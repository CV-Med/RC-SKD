import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import sys
from models.corrective_attention import Loss_compute
import os
import json


def train(student, teacher, epochs, best_acc, dataset='Brain_3'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student.to(device)
    teacher.to(device)

    optimizer = optim.Adam(student.parameters(), lr=0.001)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    if dataset == 'CIFAR100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, transform=transform)
        val_dataset = datasets.CIFAR100(root='./data', train=False, transform=transform)
        batch_size = 32
    else:
        data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))
        image_path = os.path.join(data_root, "data", dataset)
        assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
        train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=transform)
        val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=transform)
        batch_size = 16

    train_num = len(train_dataset)
    val_num = len(val_dataset)

    if dataset != 'CIFAR100':
        flower_list = train_dataset.class_to_idx
        cla_dict = dict((val, key) for key, val in flower_list.items())
        json_str = json.dumps(cla_dict, indent=3)
        with open("class_indices.json", 'w') as f:
            f.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=nw, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=nw, drop_last=True)
    print(f"Using {train_num} images for training, {val_num} images for validation.")

    for epoch in range(epochs):
        student.train()
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss_total = Loss_compute()
            loss = loss_total(teacher, student, images, labels, device)
            loss.backward()
            optimizer.step()
            train_bar.desc = "train epoch [{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        student.eval()
        correct = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = student(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels.to(device)).sum()
        val_acc = correct / len(val_loader.dataset)
        print(f"Epoch [{epoch + 1}/{epochs}], Validation Accuracy: {val_acc:.4f}")
        if best_acc <= val_acc:
            best_acc = val_acc
            model_name = f"./weights/tkd_ca_{best_acc:.3f}_model.pth"
            torch.save(student, model_name)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('./weights', exist_ok=True)
    teacher_path = 'ACI_D201_0.695_model.pth'
    teacher = torch.load(teacher_path, weights_only=False).to(device)
    student_path = 'CI_D201_0.570_model.pth'
    student = torch.load(student_path, weights_only=False).to(device)
    train(student, teacher, epochs=200, best_acc=0.0, dataset='Brain_3')
