import torch
import pandas as pd


def normalize_position(position, eps=1e-8):
    '''
    Max-Min 归一化方法, 投影到 [0, 1]
    '''
    min_vals = position.min(dim=0, keepdim=True).values
    max_vals = position.max(dim=0, keepdim=True).values
    # 避免除 0 误差
    range_vals = max_vals - min_vals
    range_vals[range_vals < eps] = 1.0  # Prevent division by zero
    
    position_norm = (position - min_vals) / range_vals

    return position_norm

class DataNormalizer:
    def __init__(self, file_list):
        self.mean = None
        self.std = None
        self._init_statistic_value(file_list)
        
    def _init_statistic_value(self, file_list):
        '''
        获取归一化的均值与方差，
        当前直接读取全量数据后计算，内存占用约 35 MB * 7 files
        扩展超大规模数据集时, 必须使用流式算法计算统计量
        '''
        # 读取全量数据
        data_stack = []
        for file_path in file_list:
            data_raw = pd.read_csv(file_path)
            data_raw = data_raw.sort_values(['DDATETIME', 'GRIDID'])

            # 保留特定的列
            features = [
                'T', 'MAXTOFDAY', 'SLP', 'RHSFC', 'V', 'RAIN01H', 'RAIN02H', 'RAIN03H', 'RAIN06H', 'RAIN24H', 
                'WSPD_X', 'WSPD_Y', 'WD3SMAXDF_X', 'WD3SMAXDF_Y', 'AIR_DENSITY'
            ] # 数据规格应写入 config
            data_array = data_raw[features].values

            tensor_data = torch.from_numpy(data_array).float()
            tensor_data = tensor_data.reshape((144, 4232, 15)) # 时间节点数 空间节点数 数据模式数 # 数据规格应写入 config

            data_stack.append(tensor_data)
        
        # 按照第一个维度（时间）拼接
        result_tensor = torch.cat(data_stack, dim=0)

        # 保留统计量
        self.mean = result_tensor.mean(dim=(0,1)) # 这里保留数据模式的差异性，数据归一化时必须分离不同的数据模式，例如气温约 10 - 20 之间，气压在 2000 左右，不可混淆
        self.std = result_tensor.std(dim=(0,1))

        # 处理标准差为0的情况，避免除零错误
        self.std = torch.where(self.std < 1e-8, torch.ones_like(self.std), self.std)

    def normalize(self, data):
        """ 归一化数据 """
        return (data - self.mean) / self.std
    
    def denormalize(self, normalized_data):
        """ 反归一化数据 """
        return normalized_data * self.std + self.mean