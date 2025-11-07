import torch
import numpy as np
from sklearn.neighbors import kneighbors_graph


def dist_to_weight(dist_matrix, sigma_2=1.0):
    """ 将距离转换为权重（与距离成反比） """
    weights_matrix = dist_matrix.clone()
    weights_matrix = weights_matrix / weights_matrix.max(1).values
    non_zero_mask = weights_matrix > 0 # 去除自环和无连接边
    weights_matrix[non_zero_mask] = torch.exp(-(weights_matrix[non_zero_mask] ** 2) / (2 * sigma_2))
    # weights_matrix[non_zero_mask] = 1.0 / (1.0 + weights_matrix[non_zero_mask])
    return weights_matrix


def matrix_normalize_by_row(matrix):
    """ 对邻接矩阵进行归一化处理 """
    # 归一化每行的权重，使每行的权重之和为1
    row_sums = matrix.sum(dim=1)
    row_sums[row_sums == 0] = 1.0  # 避免除零
    matrix = matrix / row_sums.unsqueeze(1)
    return matrix


def calculate_graph(k: int = 8) -> torch.tensor:
    print('正在计算邻接矩阵')
    coords_path = './dataset/coords_data.npy' # 应该写入 config
    coords_raw = np.load(coords_path)
    coords_raw = torch.tensor(coords_raw)

    dist_matrix = kneighbors_graph(coords_raw, n_neighbors=k, mode='distance', include_self=False)
    dist_matrix = torch.tensor(dist_matrix.toarray(), dtype=torch.float)
    
    # 转换为权重矩阵并归一化
    weights_matrix = dist_to_weight(dist_matrix)
    adj_matrix = matrix_normalize_by_row(weights_matrix)
    print('邻接矩阵计算完毕')
    return adj_matrix