"""
图像场景分类系统 - 主程序入口

本模块是整个项目的入口点，负责协调训练和评估流程。
执行顺序：
    1. 创建日志目录
    2. 加载配置
    3. 训练模型
    4. 评估模型
"""

import os
import warnings
from config import Config
from train import train
from evaluate import evaluate

warnings.filterwarnings("ignore")

def main():
    if not os.path.exists(Config.LOG_PATH):
        os.makedirs(Config.LOG_PATH)

    config = Config()
    train(config)
    evaluate(config)

if __name__ == "__main__":
    main()