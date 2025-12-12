import torch
import sys
import os
import json
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import datasets, transforms
from torchvision.models import ResNet101_Weights, ResNet50_Weights, ResNet34_Weights, ResNet18_Weights


# 剪枝卷积层
def prune_conv_layer(conv_layer, num_pruned_channels):
    weight = conv_layer.weight.data
    norms = weight.view(weight.size(0), -1).norm(2, dim = 1)
    _, indices = torch.topk(norms, k = num_pruned_channels, largest = True)
    pruned_weight = weight[indices]
    new_conv_layer = nn.Conv2d(
        in_channels = conv_layer.in_channels,
        out_channels = pruned_weight.size(0),
        kernel_size = conv_layer.kernel_size,
        stride = conv_layer.stride,
        padding = conv_layer.padding,
        bias = conv_layer.bias is not None
    )
    new_conv_layer.weight.data = pruned_weight.clone()
    if conv_layer.bias is not None:
        new_conv_layer.bias.data = conv_layer.bias.data[indices].clone()
    return new_conv_layer, indices


# 剪枝BatchNorm层
def prune_bn_layer(bn_layer, remaining_indices):
    new_bn_layer = nn.BatchNorm2d(len(remaining_indices))
    new_bn_layer.weight.data = bn_layer.weight.data[remaining_indices].clone()
    new_bn_layer.bias.data = bn_layer.bias.data[remaining_indices].clone()
    new_bn_layer.running_mean = bn_layer.running_mean[remaining_indices].clone()
    new_bn_layer.running_var = bn_layer.running_var[remaining_indices].clone()
    return new_bn_layer


# 剪枝BasicBlock
def prune_basic_block(block, prune_ratio, input_channels):
    """裁剪BasicBlock中的conv1层，并确保与残差连接兼容"""
    conv1 = block.conv1
    bn1 = block.bn1
    conv2 = block.conv2
    bn2 = block.bn2

    # 剪枝 conv1
    num_pruned_channels_conv1 = int(conv1.out_channels * prune_ratio)
    if num_pruned_channels_conv1 > 0:
        new_conv1, remaining_idx = prune_conv_layer(conv1, num_pruned_channels_conv1)
        block.conv1 = new_conv1
    else:
        remaining_idx = torch.arange(conv1.out_channels)

    # 剪枝 bn1
    block.bn1 = prune_bn_layer(bn1, remaining_idx)

    # 更新 conv2，使其输入通道与 `conv1` 的输出通道相同，输出通道数与输入通道数匹配
    new_conv2 = nn.Conv2d(
        in_channels = len(remaining_idx),  # 新的输入通道数
        out_channels = input_channels,  # 确保输出通道数与输入通道数一致
        kernel_size = conv2.kernel_size,
        stride = conv2.stride,
        padding = conv2.padding,
        bias = conv2.bias is not None
    )
    # 将原 `conv2` 裁剪后的通道权重赋值到 `new_conv2`
    new_conv2.weight.data = conv2.weight.data[:input_channels, :len(remaining_idx), :, :].clone()
    block.conv2 = new_conv2

    # 更新 bn2，使其通道数与 `conv2` 的输出通道一致
    block.bn2 = prune_bn_layer(bn2, torch.arange(input_channels))

    return block


# 剪枝模型
def prune_model(model, ratios):
    """遍历ResNet模型的各个layer，按给定的剪枝比例prune_ratios执行剪枝"""
    prune_ratios = {
        "layer1": ratios[0],
        "layer2": ratios[1],
        "layer3": ratios[2],
        "layer4": ratios[3]
    }
    input_channels = 64  # ResNet的初始输入通道数
    for name, module in model.named_children():
        if name.startswith("layer"):
            for i, block in enumerate(module):
                prune_ratio = prune_ratios.get(name, 0)
                pruned_block = prune_basic_block(block, prune_ratio, input_channels)
                module[i] = pruned_block
            input_channels *= 2  # 每一层输出通道数加倍
    return model


# 模型训练与评估
def train_eval(compressed_model, epochs, train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"use device is {device}")
    print('裁剪后的网络结构')
    print(compressed_model)
    compressed_model.to(device)
    # weights_init(compressed_model)

    loss_function = nn.CrossEntropyLoss()
    model_optimizer = optim.Adam(compressed_model.parameters(), lr = 0.0001)
    best_acc = 0.712
    # 训练生成的候选模型
    for epoch in range(epochs):
        # 训练和验证
        compressed_model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file = sys.stdout)
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
            f"Epoch [{epoch + 1}/{epochs}], Validation Accuracy: {val_acc:.3f}")
        if best_acc <= val_acc:
            best_acc = val_acc
            print(f'beast acc: {best_acc:.3f}')
            model_name = f"./Weight/CI_R50_{best_acc:.3f}_model.pth"
            torch.save(compressed_model, model_name)
    return best_acc


# 已有的目标模型stu_model（输入为超参数，输出为准确率）
def target_model(state):
    print(state)
    # 保存到 txt 文件
    output_file = "R50_state.txt"
    with open(output_file, "a") as f:
        f.write(f"{state}\n")  # 每个值占一行

    tea_model = models.resnet50(weights = ResNet50_Weights.DEFAULT)
    # 修改最后的全连接层以适应新数据集
    num_classes = 100  # 将此处改为类别数
    tea_model.fc = torch.nn.Linear(tea_model.fc.in_features, num_classes)
    stu_model = prune_model(tea_model, state)
    # 数据预处理，包括数据增强
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding = 4),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])
    # 加载 CIFAR100 训练集与验证集
    train_data = torchvision.datasets.CIFAR100(
        root = './data', train = True, transform = transform)
    val_data = torchvision.datasets.CIFAR100(
        root = './data', train = False, transform = transform)

    # # 加载 STL-10 数据集
    # train_data = datasets.STL10(root = './data', split = 'train',
    #                                transform = transform)
    # val_data = datasets.STL10(root = './data', split = 'test',
    #                               transform = transform)
    # 数据加载器
    batch_size = 32
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 2)
    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = False, num_workers = 2)

    # # 加载 Brain数据集
    # data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))
    # image_path = os.path.join(data_root, "data", "Brain_4")
    # assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # train_dataset = datasets.ImageFolder(root = os.path.join(image_path, "train"),
    #                                      transform = transform)
    # train_num = len(train_dataset)
    # flower_list = train_dataset.class_to_idx
    # cla_dict = dict((val, key) for key, val in flower_list.items())
    # json_str = json.dumps(cla_dict, indent = 3)
    # with open("calss_indices.json", 'w') as json_file:
    #     json_file.write(json_str)
    #
    # batch_size = 16
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # 线程数计算
    #
    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                            batch_size = batch_size,
    #                                            shuffle = True,
    #                                            num_workers = nw)
    # val_dataset = datasets.ImageFolder(root = os.path.join(image_path, "val"),
    #                                    transform = transform)
    # val_num = len(val_dataset)
    # val_loader = torch.utils.data.DataLoader(val_dataset,
    #                                          batch_size = batch_size,
    #                                          shuffle = False,
    #                                          num_workers = nw
    #                                          )
    # print(f"Using {train_num} images for training, {val_num} images for validation.")
    # 开始强化压缩
    reward = train_eval(stu_model, 200, train_loader, val_loader)
    # 保存到 txt 文件
    output_file = "R50_val.txt"
    with open(output_file, "a") as f:
        f.write(f"{reward}\n")  # 每个值占一行
    return reward


if __name__ == '__main__':
    # 训练环境测试
    print(target_model([0.5, 0.3, 0.35, 0.31]))
