"""
图像场景分类系统 - 预测模块

本模块用于加载训练好的模型，对新的图片进行预测，并可视化Top-N结果。
支持批量预测和结果保存。
"""

import os
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
from model import SceneClassifier
from config import Config
import torch.nn.functional as F
from torchvision import transforms


def predict_image(model, image_tensor, device, top_n=3):
    # 单张图片Top-N预测
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        top_probs, top_classes = torch.topk(probabilities, top_n, dim=1)

        # 将概率, 类别转为numpy, 因为
        return top_probs.cpu().numpy()[0], top_classes.cpu().numpy()[0]


def main():
    config = Config()
    device = config.DEVICE
    class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

    # 加载模型
    model = SceneClassifier(num_classes=6).to(device)
    model.load_state_dict(torch.load(os.path.join(config.LOG_PATH, 'best_model.pth'), map_location=device))

    # 预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 获取所有png图片
    image_paths = []
    for root, dirs, files in os.walk(Config.PRED_PATH):
        for file in files:
            image_paths.append(os.path.join(root, file))

    print(f"找到 {len(image_paths)} 张图片")

    # 随机选8张
    selected_paths = random.sample(image_paths, min(8, len(image_paths)))
    print(f"选择了: {selected_paths}")

    # 预测并收集结果
    results = []
    for path in selected_paths:
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img)
        probs, classes = predict_image(model, img_tensor, device)
        results.append((path, img, probs, classes))

    # 可视化: 2行4列
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (path, img, probs, classes) in enumerate(results):
        # 显示图片
        axes[i].imshow(img)
        axes[i].axis('off')

        # 移除所有边框
        for spine in axes[i].spines.values():
            spine.set_visible(False)

        # 构建标题: 文件名 + Top3预测
        filename = os.path.basename(path)
        title = f"{filename}\n"
        for i, (prob, cls) in enumerate(zip(probs, classes)):
            title += f"{class_names[cls]}:{prob:.4f}\n"

        axes[i].set_title(title, fontsize=9, pad=5)

    # 自动调节图片的间距, 边距等
    plt.tight_layout()
    save_path = os.path.join(config.LOG_PATH, 'predict.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()