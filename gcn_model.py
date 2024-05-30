import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats):
        """
        初始化 GCN 模型。

        参数：
        in_feats (int): 输入特征的维度。
        h_feats (int): 隐藏层特征的维度。
        """
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(h_feats, in_feats, allow_zero_in_degree=True)  # 输出特征维度与输入特征维度相同

    def forward(self, g, in_feat):
        """
        前向传播。

        参数：
        g (DGLGraph): 输入图。
        in_feat (torch.Tensor): 输入特征。

        返回：
        torch.Tensor: 输出特征。
        """
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

# def train_gcn_model(train_graphs, in_feats, h_feats, out_feats, epochs=50, learning_rate=0.01):
#     """
#     训练 GCN 模型。
#
#     参数：
#     train_graphs (list): 训练图列表。
#     in_feats (int): 输入特征的维度。
#     h_feats (int): 隐藏层特征的维度。
#     out_feats (int): 输出特征的维度。
#     epochs (int): 训练轮数。
#     learning_rate (float): 学习率。
#
#     返回：
#     GCN: 训练好的 GCN 模型。
#     """
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = GCN(in_feats, h_feats, out_feats).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     loss_fn = nn.MSELoss()
#
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         for g in train_graphs:
#             g = g.to(device)
#             logits = model(g, g.ndata['feat'])
#             optimizer.zero_grad()
#             loss = loss_fn(logits, g.ndata['feat'])
#             loss.backward()
#             # 梯度裁剪，防止梯度爆炸
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             total_loss += loss.item()
#         print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_graphs)}')
#
#     return model

def train_gcn_model(train_graphs, in_feats, h_feats, epochs=50):
    """
    训练 GCN 模型。

    参数：
    train_graphs (list): 训练图列表。
    in_feats (int): 输入特征的维度。
    h_feats (int): 隐藏层特征的维度。
    epochs (int): 训练轮数。

    返回：
    GCN: 训练好的 GCN 模型。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(in_feats, h_feats).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for g in train_graphs:
            g = g.to(device)
            logits = model(g, g.ndata['feat'])
            optimizer.zero_grad()
            loss = loss_fn(logits, g.ndata['feat'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_graphs)}')

    return model