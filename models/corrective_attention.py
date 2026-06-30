import torch
import torch.nn as nn
import torch.nn.functional as F


teacher_features = []
student_features = []
gradients = []


def channel_enhancement(Fc):
    mu = Fc.mean(dim=(2, 3), keepdim=True)
    sigma = Fc.std(dim=(2, 3), keepdim=True)
    return torch.sigmoid((Fc - mu) / (sigma + 1e-5))


def spatial_enhancement(Fc):
    spatial_mean = Fc.mean(dim=1, keepdim=True)
    return torch.sigmoid(spatial_mean)


def enhance_weights(weights):
    normalized = F.softmax(weights, dim=1)
    enhanced = torch.exp(normalized * 2.0)
    adjusted = F.softmax(enhanced, dim=1)
    return adjusted


def teacher_hook(module, input, output):
    teacher_features.append(output)


def student_hook(module, input, output):
    student_features.append(output)


def save_gradients(module, grad_input, grad_output):
    gradients.append(grad_output[0])


def _register_hooks(teacher, student):
    teacher_features.clear()
    student_features.clear()
    gradients.clear()
    if hasattr(teacher, 'features') and hasattr(teacher.features, 'denseblock4'):
        teacher.features.denseblock4.denselayer16.register_forward_hook(teacher_hook)
        teacher.features.denseblock4.denselayer16.conv2.register_full_backward_hook(save_gradients)
        if hasattr(student, 'features') and len(student.features) > 10:
            student.features[10].denselayer16.conv2.register_forward_hook(student_hook)
    elif hasattr(teacher, 'layer4'):
        teacher.layer4[-1].register_forward_hook(teacher_hook)
        teacher.layer4[-1].register_full_backward_hook(save_gradients)
        student.layer4[-1].register_forward_hook(student_hook)


class Loss_compute:
    def __init__(self, alpha=3.0, T=3.0):
        self.alpha = alpha
        self.T = T

    def __call__(self, teacher, student, images, labels, device):
        _register_hooks(teacher, student)

        teacher.eval()
        teacher_output = teacher(images)
        student_output = student(images)

        target_classes = torch.argmax(teacher_output, dim=1)
        one_hot = torch.zeros(teacher_output.size(), device=device)
        one_hot.scatter_(1, target_classes.unsqueeze(1), 1)

        teacher_features[0].retain_grad()
        teacher_output.backward(gradient=one_hot, retain_graph=True)

        weights = gradients[0]
        w_c = torch.mean(torch.abs(weights), dim=(2, 3))
        w_c = enhance_weights(w_c)

        F_initial = teacher_features[0]
        F_channl = channel_enhancement(F_initial)
        F_patch = spatial_enhancement(F_initial)
        F_pre = F_initial * F_patch * F_channl
        F_final = torch.sigmoid(w_c).unsqueeze(2).unsqueeze(3) * F_pre

        cos_sim = F.cosine_similarity(student_features[0], F_final, dim=1)
        L_CE = 1 - cos_sim.mean()

        criterion_kl = nn.KLDivLoss(reduction='sum')
        soft_log_probs = F.log_softmax(student_output / self.T, dim=1)
        soft_target = F.softmax(teacher_output / self.T, dim=1)
        L_soft = criterion_kl(soft_log_probs, soft_target)

        total_loss = self.alpha * L_CE + L_soft

        teacher_features.clear()
        student_features.clear()
        gradients.clear()

        return total_loss
