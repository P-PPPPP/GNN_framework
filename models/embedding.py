import torch
import torch.nn as nn


class Spatial_Embedding(nn.Module):
    def __init__(self, d_data, d_coords, d_model):
        super(Spatial_Embedding, self).__init__()
        # Initialize layers based on config
        self.data_embed = nn.Linear(d_data, d_model)
        self.spatial_embed = nn.Linear(d_coords, d_model)

    def forward(self, data, coords, mask):
        # 数据 embedding
        data_emb = self.data_embed(data)  # (batch_size, num_nodes, d_model)
        data_emb = torch.where(~mask.unsqueeze(-1), data_emb, torch.zeros_like(data_emb)) # 保证 mask 位置的数据为 0 
        
        # 坐标 embedding
        spatial_emb = self.spatial_embed(coords)  # (batch_size, num_nodes, d_model)

        return data_emb + spatial_emb