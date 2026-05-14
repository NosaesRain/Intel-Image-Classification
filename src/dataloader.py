"""
图像场景分类系统 - 数据加载模块

本模块负责数据集的加载、预处理和数据增强。
定义了自定义Dataset类和DataLoader创建函数.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from config import Config


class SceneDataset(Dataset):
    """
    从指定目录加载图像数据，目录结构应为：
        data_dir/
            class_name1/
                img1.jpg
                img2.jpg
                ...
            class_name2/
                img1.jpg
                img2.jpg
                ...

    Attributes:
        data_dir (str): 数据集根目录路径
        transform (callable, optional): 图像预处理/增强函数
        class_names (list): 类别名称列表，按字母顺序排序
        image_paths (list): (图像路径, 类别名称) 元组列表

    Args:
        data_dir (str): 数据集目录路径，包含按类别组织的子文件夹
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.class_names = sorted(os.listdir(data_dir))
        self.image_paths = []
        for class_name in self.class_names:
            class_dir = os.path.join(data_dir, class_name)
            self.image_paths += [(os.path.join(class_dir, img), class_name) for img in os.listdir(class_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, class_name = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.class_names.index(class_name)
        if self.transform:
            image = self.transform(image)
        return image, label


def get_data_loaders(config):
    # 数据增强预处理
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        # 数据增强
        transforms.RandomHorizontalFlip(p=0.5) if
        config.AUGMENTATION['horizontal_flip'] else transforms.Lambda(lambda x: x),
        # transforms.RandomVerticalFlip(p=0.5) if
        # config.AUGMENTATION['vertical_flip'] else transforms.Lambda(lambda x: x),
        transforms.RandomRotation(config.AUGMENTATION['rotation']),
        transforms.ColorJitter(
            brightness=config.AUGMENTATION['brightness'],
            contrast=config.AUGMENTATION['contrast']
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_train_dataset = SceneDataset(data_dir=config.TRAIN_PATH, transform=train_transform)
    train_size = int(config.TRAIN_VAL_SPLIT * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    # 生成随机划分的索引
    train_indices, val_indices = random_split(
        range(len(full_train_dataset)), # 原始索引列表
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset = SceneDataset(data_dir=config.TRAIN_PATH, transform=train_transform)
    val_dataset = SceneDataset(data_dir=config.TRAIN_PATH, transform=eval_transform)

    # 使用Subset根据索引划分数据
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    test_dataset = SceneDataset(data_dir=config.TEST_PATH, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)


    return train_loader, val_loader, test_loader