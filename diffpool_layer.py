import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv
from torch_geometric.utils import to_dense_adj, to_dense_batch


class DiffPoolLayer(nn.Module):
    def __init__(self, input_dim, num_clusters):
        """
        可微分池化层
        :param input_dim: 节点的输入特征维度
        :param num_clusters: 目标簇的数量
        """
        super(DiffPoolLayer, self).__init__()
        self.input_dim = input_dim
        self.num_clusters = num_clusters

        # 用于学习分配矩阵 S 的 GNN
        self.pool_gnn = DenseGCNConv(self.input_dim, self.num_clusters)

        # 用于处理粗化图的 GNN
        self.embed_gnn = DenseGCNConv(self.input_dim, self.input_dim)

    def forward(self, x, edge_index, batch):
        # 将稀疏的 PyG 图数据转换为稠密格式，以便进行矩阵运算
        # dense_x shape: [batch_size, max_nodes_in_batch, feature_dim]
        # mask shape: [batch_size, max_nodes_in_batch]
        dense_x, mask = to_dense_batch(x, batch)
        # adj shape: [batch_size, max_nodes_in_batch, max_nodes_in_batch]
        adj = to_dense_adj(edge_index, batch)

        # 使用池化GNN计算分配矩阵 S
        # s shape: [batch_size, max_nodes_in_batch, num_clusters]
        s = self.pool_gnn(dense_x, adj)
        s = F.softmax(s, dim=-1)

        # 使用分配矩阵 S 进行池化操作
        # x_pooled shape: [batch_size, num_clusters, feature_dim]
        x_pooled = torch.matmul(s.transpose(1, 2), dense_x)
        # adj_pooled shape: [batch_size, num_clusters, num_clusters]
        adj_pooled = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

        # --- 计算辅助损失 ---

        # 链接预测损失 (Link Prediction Loss)
        # 目标是让 S * S^T 尽可能地接近原始邻接矩阵 A
        pred_adj = torch.matmul(s, s.transpose(1, 2))
        link_loss = torch.norm(adj - pred_adj, p=2)
        link_loss = link_loss / torch.norm(adj, p=2)

        # 熵损失 (Entropy Loss)
        # 目标是让分配矩阵的概率分布更明确，减少不确定性
        entropy = (-s * torch.log(s + 1e-15)).sum(dim=-1)
        entropy_loss = entropy.mean()

        # 在粗化图上运行嵌入GNN，以获得最终的簇表示
        x_pooled_out = self.embed_gnn(x_pooled, adj_pooled)

        # 将批处理后的稠密输出还原为 PyG 的稀疏格式
        # 形状为 [batch_size * num_clusters, feature_dim]
        batch_size, num_clusters, feat_dim = x_pooled_out.shape
        x_pooled_out = x_pooled_out.view(batch_size * num_clusters, feat_dim)

        return x_pooled_out, link_loss, entropy_loss