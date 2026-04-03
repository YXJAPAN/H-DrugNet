import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool

# 从新文件中导入 DiffPoolLayer
from diffpool_layer import DiffPoolLayer


# 注意：您需要确保原始 Net.py 中的 DiagLayer 类也在此文件中，或能被正确导入
class DiagLayer(torch.nn.Module):
    def __init__(self, in_dim, num_et=1):
        super(DiagLayer, self).__init__()
        self.num_et = num_et
        self.in_dim = in_dim
        self.weight = nn.Parameter(torch.Tensor(num_et, in_dim))
        self.reset_parameters()

    def forward(self, x):
        value = x * self.weight
        return value

    def reset_parameters(self):
        self.weight.data.normal_(std=1 / (self.in_dim ** 0.5))


class A3Net_DiffPool(nn.Module):
    def __init__(self, input_dim=109, input_dim_e=243, output_dim=200, dropout=0.2, heads=10, num_drug_clusters=32):
        super(A3Net_DiffPool, self).__init__()

        # --- 交叉注意力模块 (CAM) - 保持不变 ---
        self.fc_1 = nn.Linear(input_dim, output_dim)
        self.fc_2 = nn.Linear(input_dim_e, output_dim)
        self.att = nn.TransformerEncoderLayer(d_model=output_dim, nhead=8)
        self.Att = nn.TransformerEncoder(self.att, num_layers=6)

        # --- 图注意力模块 (GAM) ---
        self.gcn1 = GATConv(input_dim, 128, heads=heads)
        self.gcn2 = GATConv(128 * heads, output_dim, heads=heads)
        self.gcn5 = GATConv(output_dim * heads, output_dim)

        # === DiffPool层 (保持不变) ===
        self.drug_diffpool = DiffPoolLayer(input_dim=output_dim, num_clusters=num_drug_clusters)
        
        # --- 关键修改点 1: 移除不再需要的融合层 ---
        # self.gam_fusion_layer = nn.Linear(output_dim * 2, output_dim)

        # 副作用的GAT层 (保持不变)
        self.gcn3 = GATConv(input_dim_e, 128, heads=heads)
        self.gcn4 = GATConv(128 * heads, output_dim, heads=heads)
        self.gcn6 = GATConv(output_dim * heads, output_dim)

        # --- 其他层 (保持不变) ---
        self.fc_g1 = nn.Linear(output_dim, output_dim)
        self.fc_g2 = nn.Linear(output_dim, output_dim)
        self.fc_g3 = nn.Linear(output_dim, output_dim)
        self.fc_g4 = nn.Linear(output_dim, output_dim)
        self.relu = nn.ReLU()
        self.norm1 = nn.LayerNorm([input_dim])
        self.norm2 = nn.LayerNorm([input_dim_e])
        self.norm3 = nn.LayerNorm([output_dim])
        self.norm4 = nn.LayerNorm([output_dim])
        self.norm_1 = nn.LayerNorm([input_dim])
        self.norm_2 = nn.LayerNorm([1280])
        self.norm_3 = nn.LayerNorm([2000])
        self.norm_e_1 = nn.LayerNorm([input_dim_e])
        self.norm_e_2 = nn.LayerNorm([1280])
        self.norm_e_3 = nn.LayerNorm([2000])
        self.diag = DiagLayer(in_dim=output_dim)

    def forward(self, data, data_e, DF=False, not_FC=True, alpha=0.15):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_e, edge_index_e = data_e.x, data_e.edge_index

        # CAM 路径 (完全不变)
        x_norm = self.norm1(x)
        x_fc = self.relu(self.fc_1(x_norm))
        x_e_norm = self.norm2(x_e)
        x_e_fc = self.relu(self.fc_2(x_e_norm))
        x_x_e = torch.cat((x_fc, x_e_fc), dim=0)
        x_x_e = self.relu(self.Att(x_x_e))
        drug_emb0, si_eff_emb0 = torch.split(x_x_e, [x.size(0), x_e.size(0)], dim=0)
        drug_emb0_pooled = global_max_pool(drug_emb0, batch)

        # GAM 路径 (药物部分被修改)
        x_gam = self.norm_1(x)
        x_gam = self.relu(self.gcn1(x_gam, edge_index))
        x_gam = self.norm_2(x_gam)
        x_gam = self.relu(self.gcn2(x_gam, edge_index))
        x_gam = self.norm_3(x_gam)
        x_gam = self.relu(self.gcn5(x_gam, edge_index)) # 这是最终的节点嵌入 [num_nodes, output_dim]

        # --- 关键修改点 2: 切换为纯 DiffPool 路径 ---
        
        # 1. 通过 DiffPool 计算分层表征
        # x_gam_pooled 是池化后的簇节点特征 [num_graphs*num_clusters, output_dim]
        # link_loss 和 entropy_loss 是辅助损失
        x_gam_pooled, link_loss, entropy_loss = self.drug_diffpool(x_gam, edge_index, batch)
        
        # 2. 为池化后的簇向量创建新的 batch 索引，以便进行全局聚合
        num_graphs = batch.max().item() + 1
        num_clusters = self.drug_diffpool.num_clusters
        new_batch = torch.arange(num_graphs, device=x.device).view(-1, 1).repeat(1, num_clusters).view(-1)
        
        # 3. 聚合所有簇的特征，得到每个图的最终表征
        # 这就是我们 GAM 路径的最终药物表征
        drug_emb_gam = global_max_pool(x_gam_pooled, new_batch) # [batch_size, output_dim]
        
        # --- 纯DiffPool路径结束 ---

        # 副作用处理 (完全不变)
        x_e_gam = self.norm_e_1(x_e)
        x_e_gam = self.relu(self.gcn3(x_e_gam, edge_index_e))
        x_e_gam = self.norm_e_2(x_e_gam)
        x_e_gam = self.relu(self.gcn4(x_e_gam, edge_index_e))
        x_e_gam = self.norm_e_3(x_e_gam)
        si_eff_emb_gam = self.relu(self.gcn6(x_e_gam, edge_index_e))

        # 特征聚合 (FAM) 路径 (基本不变)
        final_drug_emb = (1 - alpha) * drug_emb_gam + alpha * drug_emb0_pooled
        final_side_effect_emb = (1 - alpha) * si_eff_emb_gam + alpha * si_eff_emb0

        # 输出层 (完全不变)
        if not not_FC:
            final_drug_emb = self.norm3(final_drug_emb)
            final_drug_emb = self.relu(self.fc_g1(final_drug_emb))
            final_drug_emb = self.fc_g2(final_drug_emb)
            final_side_effect_emb = self.norm4(final_side_effect_emb)
            final_side_effect_emb = self.relu(self.fc_g3(final_side_effect_emb))
            final_side_effect_emb = self.fc_g4(final_side_effect_emb)

        x_ = self.diag(final_drug_emb) if DF else final_drug_emb
        xc = torch.matmul(x_, final_side_effect_emb.T)

        return xc, final_drug_emb, final_side_effect_emb, link_loss, entropy_loss