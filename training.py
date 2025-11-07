import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train(config, model, train_loader, test_loader, adj):
    """
    训练图神经网络模型（这个函数是 d 老师写的）(bug 还挺多)
    """
    # 设置设备
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 移动模型和邻接矩阵到设备
    model = model.to(device)
    adj = adj.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), 
                          lr=config['learning_rate'],
                          weight_decay=config['weight_decay'])
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # 训练历史记录
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    
    # 训练循环
    for epoch in range(config['num_epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Train]')
        
        for batch_idx, (masked_data, target_data, coordinates, mask) in enumerate(train_bar):
            # 移动数据到设备
            masked_data = masked_data.to(device)
            target_data = target_data.to(device)
            coordinates = coordinates.to(device)
            mask = mask.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(masked_data, coordinates, mask, adj)
            
            # 计算损失
            loss = criterion(output[mask,:], target_data[mask,:]) # 这里 mask 的含义：强制模型只关注被遮掩位置的数据
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（可选）
            if config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            optimizer.step()
            
            # 统计损失
            train_loss += loss.item()
            train_bar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            test_bar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Test]')
            for masked_data, target_data, coordinates, mask in test_bar:
                masked_data = masked_data.to(device)
                target_data = target_data.to(device)
                coordinates = coordinates.to(device)
                mask = mask.to(device)
                
                output = model(masked_data, coordinates, mask, adj)
                loss = criterion(output[mask,:], target_data[mask,:]) # 这里 mask 的含义：强制模型只关注被遮掩位置的数据
                test_loss += loss.item()
                test_bar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        # 计算平均测试损失
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        # 更新学习率
        scheduler.step(avg_test_loss)
        
        # 打印训练信息
        print(f'Epoch {epoch+1}/{config["num_epochs"]}, '
              f'Train Loss: {avg_train_loss:.6f}, '
              f'Test Loss: {avg_test_loss:.6f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # 保存最佳模型
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
            }, config['model_save_path'])
            print(f'Best model saved with test loss: {best_test_loss:.6f}')
    
    print(f"Training completed! Best test loss: {best_test_loss:.6f}")
    return train_losses, test_losses

