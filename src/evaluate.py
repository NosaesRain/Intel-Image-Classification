"""
图像场景分类系统 - 模型评估模块

本模块负责在测试集上评估训练好的模型性能。
生成分类报告（precision/recall/f1-score）和混淆矩阵可视化。
"""

import os
import torch
from model import SceneClassifier
from sklearn.metrics import classification_report, confusion_matrix
from dataloader import get_data_loaders
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate(config):
    """
    在测试集上评估模型性能

    执行流程：
        1. 加载训练好的最佳模型权重
        2. 遍历整个测试集进行预测
        3. 计算并打印详细的分类指标（每个类别的precision/recall/f1-score）
        4. 绘制混淆矩阵并保存

    Args:
        config (Config): 配置对象，包含路径和设备信息

    Returns:
        None，但会在logs目录生成：
            - confusion_matrix.png: 混淆矩阵可视化图像
    """
    device = config.DEVICE
    model = SceneClassifier(num_classes=6).to(device)
    model.load_state_dict(torch.load(os.path.join(config.LOG_PATH, 'best_model.pth')))
    model.eval()

    _, _, test_loader = get_data_loaders(config)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    class_names = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds).astype(int)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(config.LOG_PATH, 'confusion_matrix.png'))

