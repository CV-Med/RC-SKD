import torch
import sys
import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import datasets, transforms

DENSENET_ARCHS = {
    121: ('densenet121', models.densenet121),
    169: ('densenet169', models.densenet169),
    201: ('densenet201', models.densenet201),
}


def prune_conv1(conv_layer, prune_ratio):
    weight = conv_layer.weight.data
    num_pruned = int(conv_layer.out_channels * prune_ratio)
    norms = weight.view(weight.size(0), -1).norm(2, dim=1)
    _, indices = torch.topk(norms, k=num_pruned, largest=True)
    pruned_weight = weight[indices]
    new_conv = nn.Conv2d(
        in_channels=conv_layer.in_channels,
        out_channels=pruned_weight.size(0),
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        bias=conv_layer.bias is not None
    )
    new_conv.weight.data = pruned_weight.clone()
    if conv_layer.bias is not None:
        new_conv.bias.data = conv_layer.bias.data[indices].clone()
    return new_conv, indices


def prune_conv2(conv_layer, remaining_indices):
    weight = conv_layer.weight.data
    pruned_weight = weight[:, remaining_indices, :, :]
    new_conv = nn.Conv2d(
        in_channels=pruned_weight.size(1),
        out_channels=conv_layer.out_channels,
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        bias=conv_layer.bias is not None
    )
    new_conv.weight.data = pruned_weight.clone()
    if conv_layer.bias is not None:
        new_conv.bias.data = conv_layer.bias.data.clone()
    return new_conv, remaining_indices


def prune_bn_layer(bn_layer, remaining_indices):
    new_bn = nn.BatchNorm2d(len(remaining_indices))
    new_bn.weight.data = bn_layer.weight.data[remaining_indices].clone()
    new_bn.bias.data = bn_layer.bias.data[remaining_indices].clone()
    new_bn.running_mean = bn_layer.running_mean[remaining_indices].clone()
    new_bn.running_var = bn_layer.running_var[remaining_indices].clone()
    return new_bn


def prune_bottleneck_block(layer, prune_ratio):
    if hasattr(layer, 'conv1') and hasattr(layer, 'conv2'):
        indices = torch.arange(layer.conv1.in_channels)
        layer.norm1 = prune_bn_layer(layer.norm1, indices)
        new_conv1, indices = prune_conv1(layer.conv1, prune_ratio)
        layer.conv1 = new_conv1
        layer.norm2 = prune_bn_layer(layer.norm2, torch.arange(layer.conv1.out_channels))
        new_conv2, _ = prune_conv2(layer.conv2, indices)
        layer.conv2 = new_conv2
    return layer


def prune_denseblock(dense_block, prune_ratio):
    for name, layer in dense_block.named_children():
        if name.startswith("denselayer"):
            dense_block[name] = prune_bottleneck_block(layer, prune_ratio)
    return dense_block


def prune_densenet(model, prune_ratios):
    model_layers = list(model.features.children())
    conv0 = model_layers.pop(0)
    norm0 = model_layers.pop(0)
    relu0 = model_layers.pop(0)
    pool0 = model_layers.pop(0)
    norm5 = model_layers.pop()

    for i, dense_block in enumerate(model_layers):
        prune_ratio = prune_ratios[i // 2]
        model_layers[i] = prune_denseblock(dense_block, prune_ratio)

    classifier = model.classifier
    model.features = nn.Sequential(conv0, norm0, relu0, pool0, *model_layers, norm5)
    model.classifier = classifier
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


def target_model(state, arch=201, num_classes=100, dataset='CIFAR100', save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Pruning ratios: {state}")

    arch_name, arch_fn = DENSENET_ARCHS[arch]
    tea_model = arch_fn(weights='DEFAULT')
    tea_model.classifier = torch.nn.Linear(tea_model.classifier.in_features, num_classes)
    stu_model = prune_densenet(tea_model, state)

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
    print(target_model([0.3, 0.478, 0.47, 0.3], arch=201, dataset='CIFAR100',
                        save_path='./weights/pruned_CIFAR100_densenet201.pth'))
