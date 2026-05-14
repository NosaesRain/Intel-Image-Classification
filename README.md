# 图像场景分类系统 - Intel Image Classification

## 项目概述

本项目基于 Intel Image Classification 数据集，构建了一个能够识别山脉、森林、街道、建筑物、冰川和海洋六类场景的深度学习图像分类系统。项目采用自定义 CNN 架构，实现了完整的训练、验证、测试和预测流程，并提供了丰富的可视化分析和模型解释工具。

## 快速开始

### 环境配置

```
# 克隆项目
git clone [your-repository-url]
cd scene-classification

# 安装依赖
pip install torch torchvision tqdm matplotlib seaborn pillow scikit-learn
```



### 数据集准备

1. 从 [Kaggle Intel Image Classification](https://www.kaggle.com/puneet6060/intel-image-classification) 下载数据集
2. 将数据集按以下结构组织：

```
dataset/
├── train/
│   ├── buildings/
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── street/
├── test/
│   ├── buildings/
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── street/
└── pred/
    └── [待预测图片]
```



### 运行项目

```
python main.py
```



### 自定义CNN结构

python

```
class SceneClassifier(nn.Module):
    - Conv1: 3→32, 3x3, BN, ReLU, MaxPool
    - Conv2: 32→64, 3x3, BN, ReLU, MaxPool  
    - Conv3: 64→128, 3x3, BN, ReLU
    - AdaptiveAvgPool: 7x7
    - FC1: 128*7*7 → 128 + Dropout
    - FC2: 128 → 64 + Dropout
    - FC3: 64 → 6 (输出层)
```



**设计思考**：

- **卷积层**：逐步增加通道数（8→16→32），提取从低级到高级的特征
- **批归一化**：加速收敛，缓解梯度消失
- **Dropout**：防止过拟合，增强泛化能力
- **自适应池化**：适应不同输入尺寸，保持特征图大小一致

## 核心功能实现

### 1. 数据增强策略

训练集采用以下增强方法：

- 随机水平翻转（p=0.5）
- 随机旋转（±20度）
- 颜色抖动（亮度±0.2，对比度±0.2）

验证集和测试集仅做基础预处理。

### 2. 训练优化机制

- **早停机制**：验证集10轮不提升则停止训练
- **学习率调度**：可选ReduceLROnPlateau，验证损失不降则降低学习率
- **权重衰减**：L2正则化防止过拟合

### 3. 可视化分析

- **训练曲线**：损失和准确率变化趋势
- **混淆矩阵**：模型混淆情况分析
- **Top-3预测**：置信度最高的三个类别及其概率

## 实验结果

### 性能指标（任取一次训练结果）

| 类别      | precision | recall | f1-score | support |
| :-------- | :-------- | :----- | :------- | :------ |
| buildings | 0.81      | 0.86   | 0.83     | 437     |
| forest    | 0.95      | 0.97   | 0.96     | 474     |
| glacier   | 0.82      | 0.78   | 0.80     | 553     |
| mountain  | 0.81      | 0.81   | 0.81     | 525     |
| sea       | 0.87      | 0.86   | 0.86     | 510     |
| street    | 0.91      | 0.89   | 0.90     | 501     |

**总体准确率**: 86%

------

**项目声明**：本代码完全开源，仅用于学习交流。如有引用，请注明出处。