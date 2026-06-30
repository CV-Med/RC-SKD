import torch
import argparse
import json
import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from utils.metrics import compute_metrics, compute_flops_reduction
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='RC-SKD evaluation')
    parser.add_argument('--dataset', type=str, default='Brain_4',
                        choices=['Brain_4', 'Brain_3', 'Brain_CT', 'CIFAR100'])
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model .pth file')
    parser.add_argument('--teacher_path', type=str, default=None,
                        help='Optional teacher model for comparison')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    if args.dataset == 'CIFAR100':
        batch_size = 32
        val_dataset = datasets.CIFAR100(root='./data', train=False, transform=transform)
        num_classes = 100
    else:
        batch_size = args.batch_size
        data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))
        image_path = os.path.join(data_root, "data", args.dataset)
        assert os.path.exists(image_path), f"{image_path} path does not exist."
        val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=transform)
        num_classes = len(val_dataset.classes)

    model = torch.load(args.model_path, weights_only=False).to(device)
    model.eval()

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds, all_probs, num_classes=num_classes)
    print(f"\n{'='*40}")
    print(f"Evaluation results on {args.dataset}:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    print(f"  FPR:       {metrics['fpr']:.4f}")
    print(f"  FNR:       {metrics['fnr']:.4f}")
    print(f"{'='*40}")

    if args.teacher_path:
        teacher = torch.load(args.teacher_path, weights_only=False).to(device)
        teacher.eval()
        t_labels, t_preds, t_probs = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = teacher(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                t_labels.extend(labels.cpu().numpy())
                t_preds.extend(predicted.cpu().numpy())
                t_probs.extend(probs.cpu().numpy())
        t_metrics = compute_metrics(t_labels, t_preds, t_probs, num_classes=num_classes)
        print(f"\nTeacher comparison:")
        print(f"  Student Acc: {metrics['accuracy']:.4f} vs Teacher Acc: {t_metrics['accuracy']:.4f}")
        print(f"  Gap: {t_metrics['accuracy'] - metrics['accuracy']:.4f}")

        fr = compute_flops_reduction(model, teacher, input_shape=(1, 3, 224, 224))
        print(f"  FLOPs Reduction (Eq.18): {fr:.1f}%")


if __name__ == '__main__':
    main()
