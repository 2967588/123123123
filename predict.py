import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def accuracy_score(y_true, y_pred):
    """计算准确率"""
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    correct = torch.sum(y_true_f == y_pred_f)
    return correct.item() / y_true_f.size(0)


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


def evaluate_model(model, dataloader, device='cuda'):
    """评估模型性能"""
    model.eval()

    metrics = {
        'dice': [],
        'iou': [],
        'accuracy': []
    }

    with torch.no_grad():
        for inputs, masks in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            masks = masks.to(device)

            outputs = model(inputs)
            # 如果模型输出包含辅助分类器
            if isinstance(outputs, dict):
                outputs = outputs['out']

            preds = torch.sigmoid(outputs)
            preds_binary = (preds > 0.5).float()

            # 计算每个样本的指标
            for i in range(inputs.size(0)):
                dice = dice_coef(masks[i], preds_binary[i]).item()
                iou = iou_score(masks[i], preds_binary[i]).item()
                accuracy = accuracy_score(masks[i], preds_binary[i])
                metrics['dice'].append(dice)
                metrics['iou'].append(iou)
                metrics['accuracy'].append(accuracy)

    # 计算平均指标
    avg_dice = np.mean(metrics['dice'])
    avg_iou = np.mean(metrics['iou'])
    avg_accuracy = np.mean(metrics['accuracy'])

    print(f"Average Dice Coefficient: {avg_dice:.4f}")
    print(f"Average IoU Score: {avg_iou:.4f}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")

    return metrics


def visualize_predictions(model, dataset, num_samples=5, device='cpu'):
    """可视化模型预测结果"""
    model.eval()

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))

    with torch.no_grad():
        for i in range(num_samples):
            # 获取样本
            image, mask = dataset[i]
            image_tensor = image.unsqueeze(0).to(device)

            # 预测
            output = model(image_tensor)
            if isinstance(output, dict):
                output = output['out']
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_binary = (pred > 0.5).astype(np.float32)

            # 转换为numpy数组以便可视化
            image_np = image.permute(1, 2, 0).numpy()
            mask_np = mask.squeeze().numpy()

            # 显示原图
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')

            # 显示真实掩码
            axes[i, 1].imshow(mask_np, cmap='gray')
            axes[i, 1].set_title('True Mask')
            axes[i, 1].axis('off')

            # 显示预测结果
            axes[i, 2].imshow(pred_binary, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()