import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import sys
import os
import json
from torch.optim.lr_scheduler import StepLR

# 训练过程
def train(student, teacher, epochs, best_acc):
    # 定义优化器
    # optimizer = optim.SGD(student.parameters(), lr = 0.0001)
    optimizer = optim.Adam(student.parameters(), lr = 0.0001)
    scheduler_decay = StepLR(optimizer, step_size = 30, gamma = 0.1)
    # 数据预处理，包括数据增强
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding = 4),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])
    criterion_kl = nn.KLDivLoss(reduction = 'sum')
    # # 下载和加载训练集与验证集
    # train_data = torchvision.datasets.CIFAR100(
    #     root = './data', train = True, transform = transform)
    # val_data = torchvision.datasets.CIFAR100(
    #     root = './data', train = False, transform = transform)
    # # 加载STL-10数据集
    # train_data = datasets.STL10(root = './data', split = 'train',
    #                             transform = transform)
    # val_data = datasets.STL10(root = './data', split = 'test',
    #                           transform = transform)
    # # 数据加载器
    # batch_size = 32
    # best_acc = best_acc
    # train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 2, drop_last=True)
    # val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = False, num_workers = 2, drop_last=True)
    # 加载 Brain数据集
    data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))
    image_path = os.path.join(data_root, "data", "Brain_3")
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

    for epoch in range(epochs):
        # 训练和验证
        student.train()
        teacher.eval()  # 教师网络不进行梯度更新
        train_bar = tqdm(train_loader, file = sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            teacher_output = teacher(images)
            student_output = student(images)

            # 定义硬损失函数，采用交叉熵损失
            criterion_ce = nn.CrossEntropyLoss()
            hard_loss = criterion_ce(student_output, labels)
            # 定义软损失函数，这里使用交叉熵损失和 KL 散度损失
            soft_loss = criterion_kl(
                torch.nn.functional.log_softmax(student_output / 3.0, dim = 1),
                # torch.nn.functional.softmax(teacher_output / self.T, dim=1)
                torch.nn.functional.softmax(teacher_output, dim = 1)
            )
            loss = 0.5 * hard_loss + 0.5 * soft_loss
            loss.backward()
            optimizer.step()
            scheduler_decay.step()
            train_bar.desc = "train epoch [{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)
        # 评估模型精度
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
            f"Epoch [{epoch + 1}/{epochs}], Validation Accuracy: {val_acc:.4f}")
        if best_acc <= val_acc:
            best_acc = val_acc
            model_name = f"./Weight/TZL-D121_{best_acc:.3f}_model.pth"
            torch.save(student, model_name)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化教师网络和学生网络
    teacher_path = 'ACI_R50_0.702_model.pth'
    teacher = torch.load(teacher_path, weights_only = False)
    teacher = teacher.to(device)

    student_path = 'ACI_shufflenet_0.672_model.pth'
    student = torch.load(student_path, weights_only = False)
    student = student.to(device)
    train(student, teacher, epochs = 300, best_acc = 0.672)