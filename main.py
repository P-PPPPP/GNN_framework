import os
import glob
import random


def get_data_normalizer(data_file_list):
    from data_factory.data_normalization import DataNormalizer
    data_normalizer = DataNormalizer(data_file_list)
    return data_normalizer


def get_dataloader(configs):
    data_file_list = glob.glob(os.path.join(configs['data_dir'], '*.csv'))
    random.shuffle(data_file_list)

    # 分离训练、测试集
    num_files = len(data_file_list)
    num_train = int(0.7 * num_files)

    data_file_list_train = data_file_list[:num_train]
    data_file_list_test = data_file_list[num_train:]

    # 使用训练集数据生成一个数据归一化器
    data_normalizer = get_data_normalizer(data_file_list_train)

    # 加载 dataset
    from data_factory.dataset import MyDataset # 应统一写到文件头部
    train_dataset = MyDataset(data_file_list_train, data_normalizer, config['device'], 'Train')
    test_dataset = MyDataset(data_file_list_test, data_normalizer, config['device'], 'Test')

    # 加载 dataloader
    from torch.utils.data import DataLoader # 应统一写到文件头部
    train_loader = DataLoader(
        train_dataset,
        batch_size=configs['batch_size'],
        shuffle=True,
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
    train_loader, test_loader = get_dataloader(config)
    model = get_model(config)
    adj = get_graph(config)
    
    from training import train
    # 训练模型
    train_losses, test_losses = train(config, model, train_loader, test_loader, adj)
    

if __name__ == "__main__":
    # 配置信息
    config = {
        # 数据配置
        'data_dir': './dataset',           # 数据加载的目录，包含所有CSV文件
        'batch_size': 32,                  # 训练时的批次大小，影响内存使用和训练速度
        
        # 模型架构配置
        'num_layers': 3,                   # GNN 层数，决定网络的深度
        'num_channels': 15,                # 数据通道的数量，如气温、气压、湿度等不同气象要素
        'd_model': 128,                    # 模型隐藏层维度，决定模型的表达能力
        'd_coords': 2,                     # 坐标的维度，通常是(x,y)或(经度,纬度)
        'dropout_ratio': 0.1,              # dropout 率，防止模型过拟合的正则化参数
        'knn_k': 8,                        # 生成图邻接矩阵时，使用k近邻算法的超参数
        
        # 训练配置
        'num_epochs': 100,                 # 训练的总轮数，决定训练时间长短
        'learning_rate': 1e-4,             # 学习率，控制参数更新的步长大小
        'device': 'cuda',                  # 训练设备，'cuda'使用GPU，'cpu'使用CPU
        
        # 保存配置
        'model_save_path': 'best_model.pth' # 最佳模型保存路径，用于后续加载和推理
    }

    main(config)