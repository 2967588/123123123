import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.nn.functional as F_nn

def find_matching_mask(image_name, mask_dir):
    """根据图像名称查找匹配的mask文件，支持多种格式"""
    base_name = os.path.splitext(image_name)[0]  # 获取不带扩展名的文件名

    # 定义可能的mask文件扩展名
    possible_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']

    for ext in possible_extensions:
        mask_name = base_name + ext
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(mask_path):
            return mask_name, mask_path

    raise FileNotFoundError(f"未找到与图像 {image_name} 对应的mask文件")


class SegmentationDataset(Dataset):
    """图像分割数据集，支持多种格式的图像和mask"""

    def __init__(self, image_dir, mask_dir, image_size=512, is_train=True, augment=True):
        """
        初始化数据集

        参数:
            image_dir: 图像文件夹路径
            mask_dir: mask文件夹路径
            image_size: 图像和mask的目标尺寸
            is_train: 是否为训练集
            augment: 是否应用数据增强
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.is_train = is_train
        self.augment = augment

        # 获取图像文件列表，过滤掉隐藏文件
        self.images = [f for f in os.listdir(image_dir) if not f.startswith('.')]

        # 构建图像与mask的映射关系
        self.image_mask_map = {}
        for img_name in self.images:
            try:
                mask_name, _ = find_matching_mask(img_name, mask_dir)
                self.image_mask_map[img_name] = mask_name
            except FileNotFoundError as e:
                print(f"警告: {e}，该图像将被跳过")
                self.images.remove(img_name)

        if not self.images:
            raise ValueError("没有找到有效的图像-mask对")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = self.image_mask_map[img_name]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 加载图像和mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 假设mask是单通道的

        # 调整大小
        image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        # 转换为张量并标准化到0-1
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        # 数据增强（仅用于训练集）
        if self.is_train and self.augment:
            image, mask = self._augment(image, mask)

        return image, mask

    def _augment(self, image, mask):
        """应用数据增强"""
        # 随机水平翻转
        if random.random() > 0.6:
            image = F.hflip(image)
            mask = F.hflip(mask)

        # 随机垂直翻转
        if random.random() > 0.6:
            image = F.vflip(image)
            mask = F.vflip(mask)

        # 随机旋转
        if random.random() > 0.6:
            angle = random.choice([-90, 90, 180])
            image = F.rotate(image, angle)
            mask = F.rotate(mask, angle)

        # 随机缩放
        if random.random() > 0.6:
            scale = random.uniform(0.8, 1.2)
            new_size = [int(s * scale) for s in image.shape[1:]]
            image = F.resize(image.unsqueeze(0), new_size, interpolation=F.InterpolationMode.BILINEAR).squeeze(0)
            mask = F.resize(mask.unsqueeze(0), new_size, interpolation=F.InterpolationMode.NEAREST).squeeze(0)

        # 随机裁剪
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(int(self.image_size * 0.8), int(self.image_size * 0.8)))
        image = F.crop(image, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)

        # 调整回原始大小
        image = image.unsqueeze(0)  # 添加批次维度
        mask = mask.unsqueeze(0)

        image = F_nn.interpolate(image, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        mask = F_nn.interpolate(mask, size=(self.image_size, self.image_size), mode='nearest')

        image = image.squeeze(0)  # 移除批次维度
        mask = mask.squeeze(0)

        # 随机添加高斯噪声
        if random.random() > 0.6:
            noise = torch.randn_like(image) * 0.05
            image = image + noise
            image = torch.clamp(image, 0, 1)  # 确保像素值在0-1之间

        # 随机亮度和对比度调整
        if random.random() > 0.6:
            brightness_factor = random.uniform(0.8, 1.2)
            image = F.adjust_brightness(image, brightness_factor)

        if random.random() > 0.6:
            contrast_factor = random.uniform(0.8, 1.2)
            image = F.adjust_contrast(image, contrast_factor)

        # 随机颜色抖动
        if random.random() > 0.6:
            saturation_factor = random.uniform(0.8, 1.2)
            hue_factor = random.uniform(-0.1, 0.1)
            image = F.adjust_saturation(image, saturation_factor)
            image = F.adjust_hue(image, hue_factor)

        return image, mask


def visualize_samples(dataset, num_samples=5, title="样本可视化"):
    """可视化数据集样本"""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))
    fig.suptitle(title, fontsize=16)

    for i in range(num_samples):
        image, mask = dataset[i]

        # 转换为numpy数组以便可视化
        image_np = image.permute(1, 2, 0).numpy()
        mask_np = mask.squeeze().numpy()

        # 显示图像
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title('Image')
        axes[i, 0].axis('off')

        # 显示mask
        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title('Mask')
        axes[i, 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，为标题留出空间
    plt.show()


def create_dataloaders(train_image_dir, train_mask_dir, test_image_dir, test_mask_dir,
                       image_size=512, batch_size=8, num_workers=4):
    """创建训练集和测试集的数据加载器"""
    # 创建训练集
    train_dataset = SegmentationDataset(
        train_image_dir, train_mask_dir,
        image_size=image_size, is_train=True, augment=True
    )

    # 创建验证集（不进行数据增强）
    val_dataset = SegmentationDataset(
        test_image_dir, test_mask_dir,
        image_size=image_size, is_train=False, augment=False
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )

    return train_loader, val_loader