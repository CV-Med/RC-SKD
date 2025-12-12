import torch
import torch.nn as nn
import torch_geometric.data as geom_data

# node_cos + edge_cos

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def features_to_graph(features):
    num_nodes = features.shape[0]
    # 计算所有节点对之间的相似度
    norm_features = nn.functional.normalize(features, p=2, dim=1)
    sim_matrix = torch.mm(norm_features, norm_features.t())
    # 排除自身节点的相似度
    mask = ~torch.eye(num_nodes, dtype=torch.bool, device=device)
    edge_weights = sim_matrix[mask]
    # 生成边索引
    edge_index = torch.nonzero(mask).t().contiguous()
    return geom_data.Data(x=features, edge_index=edge_index, edge_weight=edge_weights)


class Loss_compute:
    def __init__(self):
        self.gama = 5

    def __call__(self, teacher, student, images, labels, device):
        # 前向传播
        teacher.eval()  # 教师网络不进行梯度更新
        teacher_output = teacher(images)
        student_output = student(images)

        teacher_graph = features_to_graph(teacher_output)
        student_graph = features_to_graph(student_output)

        # 计算边权重的损失
        edge_cos_sim = nn.functional.cosine_similarity(student_graph.edge_weight.unsqueeze(0), teacher_graph.edge_weight.unsqueeze(0))
        edge_weight_loss = 1 - edge_cos_sim.mean()

        # 计算节点损失
        node_cos_sim = nn.functional.cosine_similarity(student_graph.x.unsqueeze(0), teacher_graph.x.unsqueeze(0))
        node_feature_loss = 1 - node_cos_sim.mean()

        # 定义硬损失函数，采用交叉熵损失
        criterion_ce = nn.CrossEntropyLoss()
        hard_loss = criterion_ce(student_output, labels)

        loss_tal = hard_loss + self.gama * (edge_weight_loss + node_feature_loss)

        return loss_tal
