import torch
import sys
import os
import json
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet34_Weights, DenseNet121_Weights, ResNet101_Weights,  DenseNet201_Weights, DenseNet169_Weights


def train_eval(compressed_model, epochs, train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"use device is {device}")
    compressed_model.to(device)

    loss_function = nn.CrossEntropyLoss()
    model_optimizer = optim.Adam(compressed_model.parameters(), lr = 0.0001)

    # 训练生成的候选模型
    for epoch in range(epochs):
        # 训练和验证
        compressed_model.train()
        running_loss = 0.0
        best_acc = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            images = images.to(device)
            outputs = compressed_model(images)
            loss = loss_function(outputs, labels.to(device))
            loss = loss.requires_grad_(True)
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()
            running_loss += loss
            train_bar.desc = "train epoch [{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # 评估模型精度
        compressed_model.eval()
        correct = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = compressed_model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels.to(device)).sum()
        val_acc = correct / len(val_loader.dataset)
        print(
            f"Epoch [{epoch + 1}/{epochs}], Validation Accuracy: {val_acc:.4f}")
        if best_acc <= val_acc:
            best_acc = val_acc
            model_name = f"CI_R34_{best_acc:.2f}_model.pth"
            torch.save(compressed_model, model_name)
    return best_acc


# 已有的目标模型stu_model（输入为超参数，输出为准确率）
def target_model(model_path, tea_model):
    if model_path == '':
        model = tea_model
    else:
        model = torch.load(model_path, weights_only=False)
    print(model)
    # 数据预处理，包括数据增强
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding = 4),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])
    # # 下载和加载训练集与验证集
    # train_data = datasets.CIFAR100(
    #     root = './data', train = True, transform = transform, download=True)
    # val_data = datasets.CIFAR100(
    #     root = './data', train = False, transform = transform, download=True)
    # # 加载STL-10数据集
    # train_data = datasets.STL10(root = './data', split = 'train',
    #                             transform = transform)
    # val_data = datasets.STL10(root = './data', split = 'test',
    #                           transform = transform)
    # 数据加载器
    # batch_size = 32
    # train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 2)
    # val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = False, num_workers = 2)
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
    print(f"Using {train_num} images for training, {val_num} images for validation.")

    reward = train_eval(model, 100, train_loader, val_loader)

    return reward


if __name__ == '__main__':
    model_path = ''
    tea_model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
    # tea_model = models.mobilenet_v3_small(pretrained = True)
    # 修改最后的全连接层以适应新数据集
    num_classes = 4
    print(tea_model)
    # tea_model.classifier[3] = torch.nn.Linear(tea_model.classifier[3].in_features, num_classes)
    # tea_model.classifier = torch.nn.Linear(tea_model.classifier.in_features, num_classes)
    tea_model.fc = torch.nn.Linear(tea_model.fc.in_features, num_classes)
    print(tea_model)
    rewards = target_model(model_path, tea_model)