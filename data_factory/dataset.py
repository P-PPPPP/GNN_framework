
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from .data_normalization import normalize_position


class MyDataset(Dataset):
    def __init__(self, file_list, data_normalizer, device, dataset_type):
        super().__init__()
        self.file_list = file_list
        self.data_normalizer = data_normalizer
        self.data_cache = {} # 缓存的机制会让训练后期加速
        self.coords = None
        self.mask = None
        self.device = device
        print(f'{dataset_type} 数据集加载完毕, 数据文件数: {len(file_list)}')

    def __len__(self):
        # 数据总长度 = 文件数量 * 单个文件中时间节点数
        length = len(self.file_list) * 144
        return length
    
    def __getitem__(self, index):
        '''
        dataset 返回主要接口
        '''
        data_raw = self._load_data(idx=index)
        # normalize
        data_normed = self.data_normalizer.normalize(data_raw)
        masked_data, mask = self._get_data(data_normed)
        target_data = self._get_target(data_normed)
        coordinates = self._get_coordinate()

        return masked_data.to(self.device), target_data.to(self.device), coordinates.to(self.device), mask.to(self.device) # 移动数据到设备

    def _load_data(self, idx):
        # 计算文件索引和批次索引
        file_idx = idx // 144
        batch_idx = idx % 144
        # 在缓存中搜索数据，如无则从文件加载
        if file_idx in self.data_cache.keys():
            full_data = self.data_cache[file_idx]
        else:
            full_data = self._load_data_from_file(file_idx)
        return full_data[batch_idx]

    def _load_data_from_file(self, file_idx):
        # 读取文件
        file_path = self.file_list[file_idx]
        data_raw = pd.read_csv(file_path)[144:] # 忽略前 144 行无用数据

        data_raw = data_raw.sort_values(['DDATETIME', 'GRIDID'])

        # 保留特定的列
        features = [
            'T', 'MAXTOFDAY', 'SLP', 'RHSFC', 'V', 'RAIN01H', 'RAIN02H', 'RAIN03H', 'RAIN06H', 'RAIN24H', 
            'WSPD_X', 'WSPD_Y', 'WD3SMAXDF_X', 'WD3SMAXDF_Y', 'AIR_DENSITY'
        ] # 数据规格应写入 config
        data_array = data_raw[features].values
    
        tensor_data = torch.from_numpy(data_array).float()
        tensor_data = tensor_data.reshape((144, 4232, 15)) # 时间节点数 空间节点数 数据模式数, # 数据规格应写入 config

        # 保存副本至缓存
        self.data_cache[file_idx] = tensor_data
        return tensor_data

    def _get_data(self, data):
        ''' 处理输入数据: 生成随机掩码屏蔽某些空间节点的值 '''
        if self.mask is None:
            # 生成随机掩码
            num_nodes = 4232
            mask_ratio = 0.2
            num_masked = int(num_nodes * mask_ratio)
            mask_indices = torch.randperm(num_nodes)[:num_masked]
            # 创建全False的掩码，然后将选中的位置设为True
            mask = torch.zeros(num_nodes, dtype=torch.bool)
            mask[mask_indices] = True
            
            self.mask = mask
            
        # 将掩码应用于数据
        masked_data = data.clone()
        masked_data[self.mask,:] = 0.0
        return masked_data, self.mask
        

    
    def _get_target(self, data):
        # 处理标签: 完整数据作为标签
        return data
    
    def _get_coordinate(self):
        '''
        获取坐标并归一化
        '''
        if self.coords is None:
            coords_path = './dataset/coords_data.npy' # 应该写入 config
            coords_raw = np.load(coords_path)
            coords_raw = torch.tensor(coords_raw)
            coords_normed = normalize_position(coords_raw)
            # 保存备份至内存
            self.coords = coords_normed

        return self.coords
