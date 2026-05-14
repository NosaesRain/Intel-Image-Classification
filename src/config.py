"""
图像场景分类系统 - 配置参数模块

本模块负责模型的目录, 路径以及参数配置。
"""

import torch
import os


class Config:
    # 获取项目根目录
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 数据集路径
    DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
    TRAIN_PATH = os.path.join(DATASET_PATH, 'train')
    TEST_PATH = os.path.join(DATASET_PATH, 'test')
    PRED_PATH = os.path.join(DATASET_PATH, 'pred')

    # 模型保存路径
    LOG_PATH = os.path.join(BASE_DIR, 'logs')  # 日志在根目录

    # 训练参数
    BATCH_SIZE = 32
    EPOCHS = 50
    PATIENCE = 10   #早停耐心值
    # 优化器参数
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 5e-4  # 权重衰减系数 5e-4

    # 学习率调度
    USE_SCHEDULER = False  # 是否使用学习率调度器
    SCHEDULER_PATIENCE = 3  # 学习率调整的耐心值
    SCHEDULER_FACTOR = 0.5  # 学习率衰减因子

    # 训练集内部分割比例（训练:验证 = 2:1）
    TRAIN_VAL_SPLIT = 0.6667  # 2/3 用于训练，1/3 用于验证

    # 数据增强参数
    AUGMENTATION = {
        'horizontal_flip': True,
        # 'vertical_flip': False,
        'rotation': 20,
        'brightness': 0.2,
        'contrast': 0.2
    }

    # 设备配置
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'