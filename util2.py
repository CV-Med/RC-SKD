import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 存储中间层输出的列表
teacher_last = []
student_last = []
gradients = []


def enhance_weights(weights):
    # 先对权重进行softmax归一化
    normalized_weights = F.softmax(weights, dim = 1)

    # 使用指数函数增强
    enhanced_weights = torch.exp(normalized_weights * 2.0)

    # 再次归一化
    adjusted_weights = F.softmax(enhanced_weights, dim = 1)

    return adjusted_weights


def MixAttention(feature_maps):
    # 1. 计算每个通道的均值和标准差
    channel_mean = torch.mean(feature_maps, dim = (2, 3), keepdim = True)  # (B, C, 1, 1)
    channel_std = torch.std(feature_maps, dim = (2, 3), keepdim = True)  # (B, C, 1, 1)

    # 2. 基于均值和标准差计算通道重要性分数
    # 标准差大的通道包含更多区分性信息
    channel_weights = channel_std / (channel_mean + 1e-5)  # 避免除零

    # 3. 归一化权重到[0,1]区间
    channel_weights = torch.sigmoid(channel_weights)

    # 1. 计算空间维度的激活强度
    spatial_weights = torch.mean(torch.abs(feature_maps), dim = 1, keepdim = True)  # (B, 1, H, W)

    # 2. 归一化
    spatial_weights = F.normalize(spatial_weights, p = 2, dim = (2, 3), eps = 1e-12)

    # 3. 应用 Softmax 使权重和为1
    spatial_weights = F.softmax(spatial_weights.view(*spatial_weights.shape[:2], -1), dim = 2)
    spatial_weights = spatial_weights.view_as(torch.mean(feature_maps, dim = 1, keepdim = True))

    # 4. 应用权重
    weighted_features = feature_maps * channel_weights
    final_weighted = weighted_features * spatial_weights

    return final_weighted


# 定义钩子函数
def teacher_hook(module, input, output):
    teacher_last.append(output)


def student_hook(module, input, output):
    student_last.append(output)


def save_gradients(module, grad_input, grad_output):
    gradients.append(grad_output[0])


class Loss_compute:
    def __init__(self):
        self.alpha = 0.5  # 平衡软目标和硬目标的权重
        self.beta = 3.0
        self.T = 3.0

    def __call__(self, teacher, student, images, labels, device):
        # 在教师网络和学生网络的中间层注册钩子函数
        # layer4[-1]/features.denseblock4.denselayer16

        teacher.features.denseblock4.denselayer16.register_forward_hook(teacher_hook)
        student.features[10].denselayer16.conv2.register_forward_hook(student_hook)
        teacher.features.denseblock4.denselayer16.conv2.register_full_backward_hook(save_gradients)

        # teacher.layer4[-1].register_forward_hook(teacher_hook)
        # student.layer4[-1].register_forward_hook(student_hook)
        # teacher.layer4[-1].register_full_backward_hook(save_gradients)

        # 前向传播
        teacher.eval()  # 教师网络不进行梯度更新
        teacher_output = teacher(images)
        student_output = student(images)
        # 假设目标分类结果为每个样本的最大概率类别
        target_classes = torch.argmax(teacher_output, dim = 1)
        # 创建 one-hot 编码，用于计算梯度
        one_hot = torch.zeros(teacher_output.size(), device = device)
        one_hot.scatter_(1, target_classes.unsqueeze(1), 1)
        # 确保 features 张量可以保留梯度
        teacher_last[0].retain_grad()
        # 计算梯度
        teacher_output.backward(gradient = one_hot, retain_graph = True)
        # 获取梯度
        weights = gradients[0]
        weights = torch.mean(weights, dim = (2, 3))
        weights = enhance_weights(weights)
        teacher_last[0] = teacher_last[0] * weights.unsqueeze(2).unsqueeze(3)
        teacher_last[0] = MixAttention(teacher_last[0])
        # 计算注意力矫正损失
        cos_sim = F.cosine_similarity(student_last[0], teacher_last[0], dim = 1)
        att_loss = 1 - cos_sim.mean()
        # 清空中间层输出列表
        teacher_last.clear()
        student_last.clear()
        gradients.clear()

        # 定义硬损失函数，采用交叉熵损失
        criterion_ce = nn.CrossEntropyLoss()
        hard_loss = criterion_ce(student_output, labels)

        # # 定义软损失函数，这里使用交叉熵损失和 KL 散度损失
        # criterion_kl = nn.KLDivLoss(reduction = 'sum')
        # soft_loss = criterion_kl(
        #     torch.nn.functional.log_softmax(student_output / self.T, dim = 1),
        #     # torch.nn.functional.softmax(teacher_output / self.T, dim=1)
        #     torch.nn.functional.softmax(teacher_output, dim = 1)
        # )
        loss_tal = hard_loss + self.beta * att_loss

        return loss_tal
