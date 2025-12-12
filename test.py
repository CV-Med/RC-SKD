import torch
from torch.utils.data import DataLoader
import json
import os
from torchvision import transforms
from torchvision import datasets

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_path = 'ACI_D121_0.393_model.pth'
    teacher = torch.load(teacher_path, weights_only = False)
    student_path = 'CI_D121_0.725_model.pth'
    student = torch.load(student_path, weights_only=False)
    # student = torch.load(student_path, map_location = torch.device('cpu'))
    print(student)
    print(teacher)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding = 4),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])
    # # 下载和加载训练集与验证集
    # train_data = torchvision.datasets.CIFAR100(
    #     root = './data', train = True, transform = transform)
    # val_data = torchvision.datasets.CIFAR100(
    #     root = './data', train = False, transform = transform)
    # batch_size = 32
    # # train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 2, drop_last = True)
    # val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = False, num_workers = 2, drop_last = True)

    # 加载 Brain数据集
    data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))
    image_path = os.path.join(data_root, "data", "Brain_4")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root = os.path.join(image_path, "train"),
                                         transform = transform)
    train_num = len(train_dataset)
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent = 3)
    with open("calss_indices.json", 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # 线程数计算

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = batch_size,
                                               shuffle = True,
                                               num_workers = nw)
    val_dataset = datasets.ImageFolder(root = os.path.join(image_path, "val"),
                                       transform = transform)
    val_num = len(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size = batch_size,
                                             shuffle = False,
                                             num_workers = nw
                                             )

    student.eval()
    correct = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = student(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels.to(device)).sum()
    val_acc = correct / len(val_loader.dataset)
    print(
        f"Validation Accuracy: {val_acc:.4f}")