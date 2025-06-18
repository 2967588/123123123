import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def dice_coef(y_true, y_pred, smooth=1e-5):
    """计算Dice系数"""
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)


def iou_score(y_true, y_pred, smooth=1e-5):
    """计算IoU分数"""
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth + smooth)


def accuracy_score(y_true, y_pred):
    """计算准确率"""
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    correct = torch.sum(y_true_f == y_pred_f)
    return correct.item() / y_true_f.size(0)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        return 1.0 - dice_coef(y_true, y_pred, self.smooth)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25, device='cpu'):
    """训练模型并返回最佳模型"""
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice = 0.0

    # 记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': [],
        'train_iou': [],
        'val_iou': [],
        'train_accuracy': [],
        'val_accuracy': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 每个epoch都有一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练模式
                dataloader = train_loader
            else:
                model.eval()  # 评估模式
                dataloader = val_loader

            running_loss = 0.0
            running_dice = 0.0
            running_iou = 0.0
            running_accuracy = 0.0

            # 迭代数据
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
            for i, (inputs, masks) in progress_bar:
                inputs = inputs.to(device)
                masks = masks.to(device)

                # 零梯度
                optimizer.zero_grad()

                # 前向传播
                # 只有在训练时才跟踪历史
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # 如果模型输出包含辅助分类器
                    if isinstance(outputs, dict):
                        outputs = outputs['out']

                    # 计算损失
                    loss = criterion(outputs, masks)

                    # 计算指标
                    preds = torch.sigmoid(outputs)
                    preds_binary = (preds > 0.5).float()

                    batch_dice = dice_coef(masks, preds_binary)
                    batch_iou = iou_score(masks, preds_binary)
                    batch_accuracy = accuracy_score(masks, preds_binary)

                    # 只有在训练时才进行反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_dice += batch_dice.item() * inputs.size(0)
                running_iou += batch_iou.item() * inputs.size(0)
                running_accuracy += batch_accuracy * inputs.size(0)

                # 更新进度条
                progress_bar.set_description(
                    f'{phase} Loss: {loss.item():.4f} Dice: {batch_dice.item():.4f} IoU: {batch_iou.item():.4f} Accuracy: {batch_accuracy:.4f}')

            # 每个epoch的统计
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_dice = running_dice / len(dataloader.dataset)
            epoch_iou = running_iou / len(dataloader.dataset)
            epoch_accuracy = running_accuracy / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Dice: {epoch_dice:.4f} IoU: {epoch_iou:.4f} Accuracy: {epoch_accuracy:.4f}')

            # 记录历史
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_dice'].append(epoch_dice)
            history[f'{phase}_iou'].append(epoch_iou)
            history[f'{phase}_accuracy'].append(epoch_accuracy)

            # 深拷贝模型
            if phase == 'val' and epoch_dice > best_dice:
                best_dice = epoch_dice
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f'Best val Dice: {best_dice:.4f}')

        # 学习率调度
        if scheduler and phase == 'train':
            scheduler.step()

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Dice: {best_dice:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, history


def plot_training_history(history):
    """绘制训练历史"""
    plt.figure(figsize=(20, 5))

    # 绘制损失曲线
    plt.subplot(1, 4, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制Dice系数曲线
    plt.subplot(1, 4, 2)
    plt.plot(history['train_dice'], label='Train Dice')
    plt.plot(history['val_dice'], label='Validation Dice')
    plt.title('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()

    # 绘制IoU曲线
    plt.subplot(1, 4, 3)
    plt.plot(history['train_iou'], label='Train IoU')
    plt.plot(history['val_iou'], label='Validation IoU')
    plt.title('IoU Score')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 4, 4)
    plt.plot(history['train_accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()