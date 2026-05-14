"""
图像场景分类系统 - 模型定义模块

本模块定义了用于场景分类的卷积神经网络模型结构，
包含自定义CNN架构，支持6类场景分类。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneClassifier(nn.Module):
    """
    自定义CNN场景分类器

    采用3层卷积+3层全连接的结构，使用BatchNorm和Dropout防止过拟合。
    输入图像经过卷积层提取特征，通过自适应池化层调整尺寸，
    最后经过全连接层输出6类场景的logits。

    Attributes:
        conv1 (nn.Conv2d): 第一层卷积，3→8通道
        bn1 (nn.BatchNorm2d): 第一层批归一化
        pool1 (nn.MaxPool2d): 第一层最大池化
        ...

    Args:
        num_classes (int): 分类类别数(6)，
        dropout (float): Dropout概率(0.5).
    """

    def __init__(self, num_classes=6, dropout=0.5):
        super(SceneClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.fc1 = nn.Linear(in_features=128 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=num_classes)

        self.dropout = nn.Dropout(dropout)  # 添加dropout防止过拟合

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.adaptive_avg_pool(x)
        # 将卷积特征图展平，以便输入全连接层 [batch, channels, h, w] -> [batch, channels*h*w]
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

# import torch.nn as nn
# import torchvision.models as models
#
# class SceneClassifier(nn.Module):
#     def __init__(self, num_classes, dropout_rate=0.5):
#         super(SceneClassifier, self).__init__()
#         self.base_model = models.resnet18(weights=None)
#         in_features = self.base_model.fc.in_features
#         self.base_model.fc = nn.Sequential(
#             nn.Dropout(dropout_rate),
#             nn.Linear(in_features, 256),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate / 2),
#             nn.Linear(256, num_classes)
#         )
#
#     def forward(self, x):
#         return self.base_model(x)

