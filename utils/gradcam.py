import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        if self.target_layer is None:
            return
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)
        if self.gradients is None or self.activations is None:
            h, w = x.shape[2], x.shape[3]
            return np.zeros((h, w))
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def visualize_gradcam(model, image_path, target_layer, save_dir='./weights/gradcam', device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    pil_img = Image.open(image_path).convert('RGB')
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    model = model.to(device)

    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(input_tensor)

    pil_resized = pil_img.resize((224, 224))
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(pil_resized)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(pil_resized)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title('Grad-CAM')
    plt.axis('off')

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(save_dir, f'{base_name}_gradcam.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'Grad-CAM saved to {save_path}')
    return save_path
