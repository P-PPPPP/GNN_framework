import torch
import torch.nn as nn


class Layer(nn.Module):
    def __init__(self, d_in, d_out, dropout_ratio=0.1):
        super(Layer, self).__init__()
        '''
        单层图信息汇聚
        '''
        self.linear = nn.Linear(d_in, d_out) # 参数矩阵 xW + b
        self.activation = nn.ReLU() # 激活函数
        self.norm = nn.LayerNorm(d_out)  # 层归一化
        # 考虑输入输出不同时的残差连接
        if d_in != d_out:
            self.shortcut = nn.Linear(d_in, d_out)
        else:
            self.shortcut = nn.Identity()
        self.dropout = nn.Dropout(dropout_ratio) # 这个超参数待优化

    def forward(self, x, adj):
        '''
        顺序对性能有影响
        '''
        # graph convolution
        shortcut = self.shortcut(x)
        # 空间图卷积: A(xW + b)
        h = self.linear(x)
        h = torch.matmul(adj, h)
        h = self.norm(h)
        # 残差连接：f(x) + x, f(x) 只学习 输出 y 和 输入 x 之间的差值
        h = h + shortcut
        h = self.activation(h)
        h = self.dropout(h)
        return h


class GNNModel(nn.Module):
    def __init__(self, embedding_module, d_model, num_channels, num_layers, dropout_ratio=0.1):
        super(GNNModel, self).__init__()
        '''
        embedding_module: 数据 embedding 模块
        d_model: embedding 维度
        num_channels: 数据模式数量
        num_layers: 堆叠 layer 层数
        '''
        self.embedding_module = embedding_module
        # 堆叠 layer
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Layer(d_model, d_model, dropout_ratio))
        # 预测器：转换至数据模式数量
        self.predictor = nn.Linear(d_model, num_channels)

    def forward(self, data: torch.tensor, adj_matrix: torch.tensor):
        # 数据嵌入
        h = self.embedding_module(data)
        # 图网络信息汇聚
        for layer in self.layers:
            h = layer(h, adj_matrix)
        # 预测节点数据
        y = self.predictor(h)
        return y