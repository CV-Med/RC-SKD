import os
import argparse
import torch
import torchvision.models as models
from torchvision.models import (
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights,
    DenseNet121_Weights, DenseNet169_Weights, DenseNet201_Weights
)
from models.lpl_rl import PruningEnv
from models.tkd_ca import train as train_tkd
from models.dkd_cg import train as train_dkd
from utils.trainer import train_eval


TEACHER_ARCH = {
    'resnet18': ('resnet', 18, models.resnet18, 'fc'),
    'resnet34': ('resnet', 34, models.resnet34, 'fc'),
    'resnet50': ('resnet', 50, models.resnet50, 'fc'),
    'resnet101': ('resnet', 101, models.resnet101, 'fc'),
    'densenet121': ('densenet', 121, models.densenet121, 'classifier'),
    'densenet169': ('densenet', 169, models.densenet169, 'classifier'),
    'densenet201': ('densenet', 201, models.densenet201, 'classifier'),
}

DATASET_CLASSES = {
    'Brain_4': 4,
    'Brain_3': 3,
    'Brain_CT': 4,
    'CIFAR100': 100,
}


def pretrain_teacher(dataset, teacher_name, num_classes, epochs):
    family, arch_num, arch_fn, head_attr = TEACHER_ARCH[teacher_name]
    model = arch_fn(weights='DEFAULT')
    head = getattr(model, head_attr)
    setattr(model, head_attr, torch.nn.Linear(head.in_features, num_classes))

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])
    if dataset == 'CIFAR100':
        train_data = datasets.CIFAR100(root='./data', train=True, transform=transform)
        val_data = datasets.CIFAR100(root='./data', train=False, transform=transform)
        batch_size = 32
    else:
        data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))
        image_path = os.path.join(data_root, "data", dataset)
        train_data = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=transform)
        val_data = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=transform)
        batch_size = 16
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)

    save_path = f"./weights/teacher_{dataset}_{teacher_name}.pth"
    best_acc = train_eval(model, epochs, train_loader, val_loader, save_path=save_path)
    print(f"[Pretrain] Teacher fine-tuned, best acc: {best_acc:.4f}, saved to {save_path}")
    return save_path


def prune_with_LPL_RL(dataset, teacher_name, num_classes, prune_epochs):
    from stable_baselines3 import DDPG
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.env_util import make_vec_env

    family, arch_num, _, _ = TEACHER_ARCH[teacher_name]
    if family == 'resnet':
        from models.cut_res import target_model as prune_fn
    else:
        from models.cut_dens import target_model as prune_fn

    save_path = f"./weights/pruned_{dataset}_{teacher_name}.pth"
    env = make_vec_env(
        lambda: PruningEnv(prune_fn, lam=1000, arch=arch_num,
                          num_classes=num_classes, dataset=dataset,
                          save_path=save_path),
        n_envs=1
    )
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=torch.zeros(n_actions).numpy(),
        sigma=0.1 * torch.ones(n_actions).numpy()
    )
    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=1000)
    model.save(f"DDPG_{dataset}_{teacher_name}.zip")
    print(f"[LPL-RL] Agent saved. Best student model -> {save_path}")
    return save_path


def distill_with_TKD_CA(teacher_path, student_path, epochs, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = torch.load(teacher_path, weights_only=False).to(device)
    student = torch.load(student_path, weights_only=False).to(device)
    train_tkd(student, teacher, epochs=epochs, best_acc=0.0, dataset=dataset)


def distill_with_DKD_CG(teacher_path, student_path, epochs, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = torch.load(teacher_path, weights_only=False).to(device)
    student = torch.load(student_path, weights_only=False).to(device)
    train_dkd(student, teacher, epochs=epochs, best_acc=0.0, dataset=dataset)


def main():
    parser = argparse.ArgumentParser(description='RC-SKD pipeline')
    parser.add_argument('--dataset', type=str, default='Brain_4',
                        choices=['Brain_4', 'Brain_3', 'Brain_CT', 'CIFAR100'])
    parser.add_argument('--teacher', type=str, default='resnet34',
                        choices=list(TEACHER_ARCH.keys()))
    parser.add_argument('--stage', type=str, default='all',
                        choices=['pretrain', 'prune', 'tkd', 'dkd', 'all'])
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--prune_epochs', type=int, default=50)
    parser.add_argument('--distill_epochs', type=int, default=200)
    parser.add_argument('--teacher_pth', type=str, default=None,
                        help='Path to pretrained teacher.')
    parser.add_argument('--student_pth', type=str, default=None,
                        help='Path to pruned student.')
    args = parser.parse_args()

    num_classes = DATASET_CLASSES[args.dataset]
    os.makedirs('./weights', exist_ok=True)

    student_pth = args.student_pth
    teacher_pth = args.teacher_pth

    if teacher_pth is None:
        teacher_pth = f"./weights/teacher_{args.dataset}_{args.teacher}.pth"

    if args.stage in ('pretrain', 'all'):
        if not os.path.exists(teacher_pth):
            print("=" * 60)
            print("Stage 0: Teacher Fine-Tuning")
            print("=" * 60)
            teacher_pth = pretrain_teacher(args.dataset, args.teacher, num_classes, args.pretrain_epochs)
        else:
            print(f"[Skip] Teacher already exists at {teacher_pth}")

    if args.stage in ('prune', 'all'):
        print("=" * 60)
        print("Stage 1: LPL-RL — Reinforcement Learning Pruning")
        print("=" * 60)
        student_pth = prune_with_LPL_RL(args.dataset, args.teacher, num_classes, args.prune_epochs)

    if args.stage in ('tkd', 'all'):
        if student_pth and os.path.exists(student_pth) and os.path.exists(teacher_pth):
            print("=" * 60)
            print("Stage 2: TKD-CA — Targeted Knowledge Distillation")
            print("=" * 60)
            distill_with_TKD_CA(teacher_pth, student_pth, args.distill_epochs, args.dataset)
        else:
            print(f"[SKIP] TKD-CA: teacher ({teacher_pth}) or student ({student_pth}) not found.")

    if args.stage in ('dkd', 'all'):
        if student_pth and os.path.exists(student_pth) and os.path.exists(teacher_pth):
            print("=" * 60)
            print("Stage 3: DKD-CG — Differential Knowledge Distillation")
            print("=" * 60)
            distill_with_DKD_CG(teacher_pth, student_pth, args.distill_epochs, args.dataset)
        else:
            print(f"[SKIP] DKD-CG: teacher ({teacher_pth}) or student ({student_pth}) not found.")

    print("RC-SKD pipeline complete.")


if __name__ == '__main__':
    main()
