import os
import glob


def get_data_normalizer(data_file_list):
    from data_factory.data_normalization import DataNormalizer
    data_normalizer = DataNormalizer(data_file_list)
    return data_normalizer


def get_dataloader(configs):
    data_file_list = glob.glob(os.path.join(configs['data_dir'], '*.csv'))
    data_file_list.sort()

    # 分离训练、测试集
    num_files = len(data_file_list)
    num_train = int(0.7 * num_files)

    data_file_list_train = data_file_list[:num_train]
    data_file_list_test = data_file_list[num_train:]

    # 使用训练集数据生成一个数据归一化器
    data_normalizer = get_data_normalizer(data_file_list_train)

    # 加载 dataset
    from data_factory.dataset import MyDataset # 应统一写到文件头部
    train_dataset = MyDataset(data_file_list_train, data_normalizer)
    test_dataset = MyDataset(data_file_list_test, data_normalizer)

    # 加载 dataloader
    from torch.utils.data import DataLoader # 应统一写到文件头部
    train_loader = DataLoader(
        train_dataset,
        batch_size=configs['batch_size'],
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=configs['batch_size'],
        shuffle=True
    )

    return train_loader, test_loader


def get_model(config):
    # embeddding module
    from models.embedding import Spatial_Embedding # 应统一写到文件头部
    embedding_module = Spatial_Embedding(
        d_data=config['num_channels'],
        d_coords=config['d_coords'],
        d_model=config['d_model']
    )
    
    # model backbone
    from models.gnn import GNNModel # 应统一写到文件头部
    model = GNNModel(
        embedding_module,
        d_model=config['d_model'],
        num_channels=config['num_channels'],
        num_layers=config['num_layers'],
        dropout_ratio=config['dropout_ratio']
    )
    return model


def get_graph(config):
    from data_factory.graph_process import calculate_graph
    adj = calculate_graph(k=config['knn_k'])
    return adj


def main(config):
    train_dataset, test_dataset = get_dataloader(config)
    model = get_model(config)
    adj = get_graph(config)
    

if __name__ == "__main__":
    # 配置信息
    config = {
        'data_dir': './dataset', # 数据加载的目录
        'batch_size': 8,
        'num_layers': 6, # GNN 层数
        'num_channels': 15, # 数据模式的数量，气温、气压、湿度...
        'd_model': 512, # 模型隐藏层维度
        'd_coords': 2, # 坐标的维度 x, y or 经度, 维度
        'dropout_ratio': 0.1, # dropout 率，反正模型过拟合的东西
        'knn_k': 8 # 生成图邻接矩阵时，使用 k 邻近算法的超参数
    }

    main(config)