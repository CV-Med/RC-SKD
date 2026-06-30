import torch
import sys
import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import datasets, transforms

RESNET_ARCHS = {
    18: ('resnet18', models.resnet18),
    34: ('resnet34', models.resnet34),
    50: ('resnet50', models.resnet50),
    101: ('resnet101', models.resnet101),
}


def prune_conv_layer(conv_layer, num_pruned_channels):
    weight = conv_layer.weight.data
    norms = weight.view(weight.size(0), -1).norm(2, dim=1)
    _, indices = torch.topk(norms, k=num_pruned_channels, largest=True)
    pruned_weight = weight[indices]
    new_conv_layer = nn.Conv2d(
        in_channels=conv_layer.in_channels,
        out_channels=pruned_weight.size(0),
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        bias=conv_layer.bias is not None
    )
    new_conv_layer.weight.data = pruned_weight.clone()
    if conv_layer.bias is not None:
        new_conv_layer.bias.data = conv_layer.bias.data[indices].clone()
    return new_conv_layer, indices


def prune_bn_layer(bn_layer, remaining_indices):
    new_bn_layer = nn.BatchNorm2d(len(remaining_indices))
    new_bn_layer.weight.data = bn_layer.weight.data[remaining_indices].clone()
    new_bn_layer.bias.data = bn_layer.bias.data[remaining_indices].clone()
    new_bn_layer.running_mean = bn_layer.running_mean[remaining_indices].clone()
    new_bn_layer.running_var = bn_layer.running_var[remaining_indices].clone()
    return new_bn_layer


def prune_basic_block(block, prune_ratio, input_channels):
    conv1 = block.conv1
    bn1 = block.bn1
    conv2 = block.conv2
    bn2 = block.bn2

    num_pruned = int(conv1.out_channels * prune_ratio)
    if num_pruned > 0:
        new_conv1, remaining_idx = prune_conv_layer(conv1, num_pruned)
        block.conv1 = new_conv1
    else:
        remaining_idx = torch.arange(conv1.out_channels)

    block.bn1 = prune_bn_layer(bn1, remaining_idx)
    new_conv2 = nn.Conv2d(
        in_channels=len(remaining_idx),
        out_channels=input_channels,
        kernel_size=conv2.kernel_size,
        stride=conv2.stride,
        padding=conv2.padding,
        bias=conv2.bias is not None
    )
    new_conv2.weight.data = conv2.weight.data[:input_channels, :len(remaining_idx), :, :].clone()
    block.conv2 = new_conv2
    block.bn2 = prune_bn_layer(bn2, torch.arange(input_channels))
    return block


def prune_model(model, ratios):
    prune_ratios = {
        "layer1": ratios[0],
        "layer2": ratios[1],
        "layer3": ratios[2],
        "layer4": ratios[3]
    }
    input_channels = 64
    for name, module in model.named_children():
        if name.startswith("layer"):
            for i, block in enumerate(module):
                pruned_block = prune_basic_block(block, prune_ratios.get(name, 0), input_channels)
                module[i] = pruned_block
            input_channels *= 2
    return model


def train_eval(compressed_model, epochs, train_loader, val_loader, device, save_path=None):
    print('Pruned model structure:')
    print(compressed_model)
    compressed_model.to(device)

    loss_function = nn.CrossEntropyLoss()
    model_optimizer = optim.Adam(compressed_model.parameters(), lr=0.01)
    best_acc = 0.0

    for epoch in range(epochs):
        compressed_model.train()
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            images = images.to(device)
            outputs = compressed_model(images)
            loss = loss_function(outputs, labels.to(device))
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()
            train_bar.desc = "train epoch [{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        compressed_model.eval()
        correct = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = compressed_model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels.to(device)).sum()
        val_acc = correct / len(val_loader.dataset)
        print(f"Epoch [{epoch + 1}/{epochs}], Validation Accuracy: {val_acc:.3f}")
        if best_acc <= val_acc:
            best_acc = val_acc
            print(f'best acc: {best_acc:.3f}')
            if save_path:
                torch.save(compressed_model, save_path)
                print(f'Model saved to {save_path}')
    return best_acc


def target_model(state, arch=50, num_classes=100, dataset='CIFAR100', save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Pruning ratios: {state}")

    arch_name, arch_fn = RESNET_ARCHS[arch]
    tea_model = arch_fn(weights='DEFAULT')
    tea_model.fc = torch.nn.Linear(tea_model.fc.in_features, num_classes)
    stu_model = prune_model(tea_model, state)

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

    reward = train_eval(stu_model, 50, train_loader, val_loader, device, save_path=save_path)
    return reward


if __name__ == '__main__':
    os.makedirs('./weights', exist_ok=True)
    print(target_model([0.5, 0.3, 0.35, 0.31], arch=50, dataset='CIFAR100',
                        save_path='./weights/pruned_CIFAR100_resnet50.pth'))
