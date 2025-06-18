import os
import torch
import matplotlib.pyplot as plt
from dateset import create_dataloaders, visualize_samples
from model import create_deeplab_model
from train import train_model, plot_training_history
from predict import evaluate_model, visualize_predictions
import ssl
import torch.nn as nn
import torch.nn.functional as F

ssl._create_default_https_context = ssl._create_unverified_context


# 实现Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 确保输入经过sigmoid转换为概率
        inputs = torch.sigmoid(inputs)

        # 展平输入和目标张量
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # 计算交集
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice  # 返回Dice Loss


# 组合损失函数：BCE Loss + Dice Loss
class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1e-5):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.bce_weight * bce + self.dice_weight * dice


def main():
    # 配置 matplotlib 字体
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 设置数据集路径（请根据实际情况修改）
    train_image_dir = r"C:\Users\27941\Desktop\大作业\1\kmms_training\images"
    train_mask_dir = r"C:\Users\27941\Desktop\大作业\1\kmms_training\masks"
    test_image_dir = r"C:\Users\27941\Desktop\大作业\1\kmms_test\images"
    test_mask_dir = r"C:\Users\27941\Desktop\大作业\1\kmms_test\masks"

    # 检查数据集路径是否存在
    for dir_path in [train_image_dir, train_mask_dir, test_image_dir, test_mask_dir]:
        if not os.path.exists(dir_path):
            print(f"错误: 目录 {dir_path} 不存在")
            return

    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader = create_dataloaders(
        train_image_dir, train_mask_dir, test_image_dir, test_mask_dir,
        image_size=512, batch_size=8, num_workers=4
    )
    for inputs, masks in train_loader:
        print(f"输入数据形状: {inputs.shape}")  # 应该显示 [batch_size, 1, height, width]
        print(f"掩码数据形状: {masks.shape}")

    # 可视化样本
    print("可视化样本...")
    visualize_samples(train_loader.dataset, num_samples=3, title="训练集样本")

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型
    print("创建模型...")
    model = create_deeplab_model(num_classes=1, pretrained=True)
    model = model.to(device)

    # 定义损失函数和优化器
    # 使用组合损失函数，这里BCE和Dice的权重各为1.0，你可以根据需要调整
    criterion = CombinedLoss(bce_weight=1.0, dice_weight=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # 训练模型
    print("开始训练模型...")
    num_epochs = 50
    best_model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=num_epochs, device=device
    )

    # 绘制训练历史，需要确保train_model函数返回的history包含组合损失
    plot_training_history(history)

    # 评估模型
    print("评估模型...")
    metrics = evaluate_model(best_model, val_loader, device=device)

    # 可视化预测结果
    print("可视化预测结果...")
    visualize_predictions(best_model, val_loader.dataset, num_samples=5, device=device)

    # 保存模型
    model_save_path = 'deeplabv3plus_model_new.pth'
    torch.save(best_model.state_dict(), model_save_path)
    print(f"模型已保存到: {model_save_path}")


if __name__ == "__main__":
    main()