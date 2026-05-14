"""
图像场景分类系统 - 训练模块

本模块负责模型的训练流程，包含单epoch训练/验证函数和主训练循环。
实现了早停机制、学习率调度、模型保存和训练曲线可视化。
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import SceneClassifier
import os
from dataloader import get_data_loaders
import matplotlib.pyplot as plt
import PIL as Image
import sys


def train_one_epoch(model, data_loader, optimizer, criterion, device):
    """
    训练模型一个epoch

    遍历整个训练集一次，执行前向传播、损失计算、反向传播和参数更新。

    Args:
        model (nn.Module): 要训练的模型
        data_loader (DataLoader): 训练数据加载器
        optimizer (Optimizer): 优化器
        criterion (nn.Module): 损失函数
        device (torch.device): 计算设备

    Returns:
        tuple: (平均损失, 准确率)
            - avg_loss (float): 本epoch的平均损失
            - accuracy (float): 本epoch的训练准确率 (0-1之间)
    """
    model.train()
    total_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for inputs, labels in tqdm(data_loader, desc="Training", leave=True):
        # 将数据移到指定设备
        inputs, labels = inputs.to(device), labels.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 统计损失与准确率
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct_preds += torch.sum(preds == labels).item()
        total_preds += labels.size(0)

    return total_loss / len(data_loader), correct_preds / total_preds


def val_one_epoch(model, data_loader, criterion, device):
    """
        在验证集上评估模型一个epoch

        遍历整个验证集一次，计算模型在验证数据上的损失和准确率。
        与训练模式不同，验证模式不进行梯度计算和参数更新。

        Args:
            model (nn.Module): 要评估的模型
            data_loader (DataLoader): 验证数据加载器
            criterion (nn.Module): 损失函数（通常与训练时相同）
            device (torch.device): 计算设备（cuda/cpu）

        Returns:
            tuple: (avg_loss, accuracy)
                - avg_loss (float): 验证集上的平均损失
                - accuracy (float): 验证集上的准确率，范围[0, 1]
        """

    model.eval()
    total_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Validating", leave=True):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels).item()
            total_preds += labels.size(0)

    return total_loss / len(data_loader), correct_preds / total_preds


def train(config):
    """
    完整的模型训练流程

    执行完整的训练循环，包括：
        1. 初始化模型、优化器、损失函数
        2. 创建数据加载器
        3. 每个epoch执行训练和验证
        4. 早停机制监控
        5. 保存最佳模型
        6. 绘制训练曲线

    Args:
        config (Config): 配置对象，包含所有超参数和路径设置
            - DEVICE: 计算设备
            - LEARNING_RATE: 学习率
            - WEIGHT_DECAY: 权重衰减系数
            - EPOCHS: 最大训练轮数
            - LOG_PATH: 日志保存路径
            - USE_SCHEDULER: 是否使用学习率调度器
            - 以及其他训练相关参数

    Returns:
        None，但在logs目录生成以下文件：
            - best_model.pth: 验证集上表现最好的模型权重
            - Train&Val_Loss.png: 训练和验证损失曲线图
            - Train&Val_Accuracy.png: 训练和验证准确率曲线图（标出最佳模型位置）
    """

    device = config.DEVICE
    print(f"使用设备: {device}")

    # 创建模型并移到指定设备
    model = SceneClassifier(num_classes=6).to(device)

    # 优化器：Adam + 权重衰减（L2正则化）
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # 损失函数：交叉熵（内部包含softmax）
    criterion = nn.CrossEntropyLoss()

    # 数据加载
    train_loader, val_loader, test_loader = get_data_loaders(config)
    print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")

    # 训练记录
    train_losses, train_accuracies = [], []  # 记录训练损失和准确率
    val_losses, val_accuracies = [], []  # 记录验证损失和准确率

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # 监控指标越小越好（验证损失）
        factor=config.SCHEDULER_FACTOR,  # 学习率衰减因子
        patience=config.SCHEDULER_PATIENCE  # 容忍轮数
    )

    # 早停机制
    best_val_accuracy = 0.0  # 最佳验证准确率
    best_epoch = 0  # 最佳模型所在轮数
    patience_counter = 0  # 早停计数器
    # 从配置中获取早停耐心值，默认为10
    early_stop_patience = getattr(config, 'EARLY_STOPPING_PATIENCE', 10)

    # 训练主循环
    for epoch in range(config.EPOCHS):
        # 训练一个epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # 验证一个epoch
        val_loss, val_acc = val_one_epoch(
            model, val_loader, criterion, device
        )

        # 记录历史数据
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # 学习率调度
        if config.USE_SCHEDULER:
            scheduler.step(val_loss)  # 根据验证损失调整学习率
            current_lr = optimizer.param_groups[0]['lr']
            print(f"当前学习率: {current_lr:.2e}")

        # 打印当前epoch结果
        print(
            f"Epoch [{epoch + 1}/{config.EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # 模型保存与早停检查
        if val_acc > best_val_accuracy:
            # 发现更好的模型
            best_val_accuracy = val_acc
            best_epoch = epoch + 1

            # 保存模型权重（只保存state_dict，不保存整个模型）
            save_path = os.path.join(config.LOG_PATH, 'best_model.pth')
            torch.save(model.state_dict(), save_path)

            patience_counter = 0
            print(f"Better model with val_acc: {val_acc:.4f}")
        else:
            # 验证准确率没有提升
            patience_counter += 1
            print(f"Patience counter: {patience_counter}")

            # 检查是否达到早停条件
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # 加载最佳模型进行最终验证
    model.load_state_dict(torch.load(os.path.join(config.LOG_PATH, 'best_model.pth')))
    val_loss, val_acc = val_one_epoch(model, val_loader, criterion, device)

    print(f"\nFinal val Performance | Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
    print(f"Best model found at epoch {best_epoch} with val accuracy: {best_val_accuracy:.4f}")

    # 绘制训练曲线
    epochs_range = range(1, len(train_losses) + 1)

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_losses, label="Train Loss", color="blue")
    plt.plot(epochs_range, val_losses, label="Val Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Train & Val Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(config.LOG_PATH, 'Train&Val_Loss.png'))
    plt.close()

    # 绘制准确率曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_accuracies, label="Train Accuracy", color="blue")
    plt.plot(epochs_range, val_accuracies, label="Val Accuracy", color="orange")
    # 用红点标记最佳模型的位置
    plt.scatter(
        best_epoch, best_val_accuracy,
        color='red', s=100, zorder=5,
        label=f'Best Model (Epoch {best_epoch})'
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Train & Val Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(config.LOG_PATH, 'Train&Val_Accuracy.png'))
    plt.close()

    print("Training complete.")