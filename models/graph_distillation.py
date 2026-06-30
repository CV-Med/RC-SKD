import torch
import torch.nn as nn
import torch.nn.functional as F


DATASET_SIZES = {
    'Brain_4': 7023,
    'Brain_3': 3064,
    'Brain_CT': 4200,
    'CIFAR100': 50000,
}


def build_graph(features):
    num_nodes = features.shape[0]
    norm_feat = F.normalize(features, p=2, dim=1)
    sim_matrix = torch.mm(norm_feat, norm_feat.t())
    mask = ~torch.eye(num_nodes, dtype=torch.bool, device=features.device)
    edge_weight = sim_matrix[mask]
    return edge_weight


class ContinuousMemoryBank:
    def __init__(self, dataset_size, feat_dim, device):
        self.memory_T = torch.zeros(dataset_size, feat_dim, device=device)
        self.memory_S = torch.zeros(dataset_size, feat_dim, device=device)
        self.ptr = 0
        self.device = device

    def update(self, logits_T, logits_S):
        batch_size = logits_T.shape[0]
        end = min(self.ptr + batch_size, self.memory_T.shape[0])
        actual_size = end - self.ptr
        self.memory_T[self.ptr:end] = logits_T[:actual_size].detach()
        self.memory_S[self.ptr:end] = logits_S[:actual_size].detach()
        self.ptr = (self.ptr + actual_size) % self.memory_T.shape[0]

    def get_global_features(self, logits_T, logits_S):
        vT_global = torch.cat([logits_T, self.memory_T], dim=0)
        vS_global = torch.cat([logits_S, self.memory_S.detach()], dim=0)
        return vT_global, vS_global


class Loss_compute:
    def __init__(self, beta=5.0, T=3.0, dataset_name='CIFAR100'):
        self.beta = beta
        self.T = T
        self.dataset_name = dataset_name
        self.memory_bank = None

    def __call__(self, teacher, student, images, labels, device):
        teacher.eval()
        with torch.no_grad():
            teacher_output = teacher(images)
        student_output = student(images)

        if self.memory_bank is None:
            feat_dim = teacher_output.shape[1]
            ds_size = DATASET_SIZES.get(self.dataset_name, 5000)
            self.memory_bank = ContinuousMemoryBank(ds_size, feat_dim, device)

        probs_T = F.softmax(teacher_output, dim=1)
        probs_S = F.softmax(student_output, dim=1)

        vT_global, vS_global = self.memory_bank.get_global_features(probs_T, probs_S)

        edge_weight_T = build_graph(vT_global)
        edge_weight_S = build_graph(vS_global)

        edge_cos = F.cosine_similarity(
            edge_weight_S.unsqueeze(0), edge_weight_T.unsqueeze(0)
        )
        edge_loss = 1 - edge_cos.mean()

        node_cos = F.cosine_similarity(
            vS_global.unsqueeze(0), vT_global.unsqueeze(0)
        )
        node_loss = 1 - node_cos.mean()

        criterion_kl = nn.KLDivLoss(reduction='sum')
        soft_log_probs = F.log_softmax(student_output / self.T, dim=1)
        soft_target = F.softmax(teacher_output / self.T, dim=1)
        L_soft = criterion_kl(soft_log_probs, soft_target)

        total_loss = L_soft + self.beta * (edge_loss + node_loss)

        self.memory_bank.update(probs_T, probs_S)

        return total_loss
