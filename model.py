import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# SE 注意力模块
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

def create_deeplab_model(num_classes=1, pretrained=True, in_channels=3):
    """创建DeepLabv3+模型，确保正确处理输入和输出通道数"""
    # 使用EfficientNet-B4作为主干网络
    if pretrained:
        backbone = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    else:
        backbone = models.efficientnet_b4(weights=None)

    # 修改第一层卷积以适应输入通道数
    if in_channels != 3:
        backbone.features[0][0] = nn.Conv2d(
            in_channels, 48, kernel_size=3, stride=2, padding=1, bias=False
        )

    # 移除最后的全连接层和池化层
    encoder = nn.Sequential(*list(backbone.features.children()))

    # 明确指定编码器的输出通道数
    encoder_channels = 1792  # EfficientNet-B4的最后一层输出通道数是1792

    # 添加 SE 注意力模块
    encoder.add_module('se_layer', SELayer(encoder_channels))

    # 创建DeepLabv3+模型
    class DeepLabv3Plus(nn.Module):
        def __init__(self, encoder, decoder, num_classes):
            super(DeepLabv3Plus, self).__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.num_classes = num_classes

        def forward(self, x):
            # 编码器前向传播
            features = self.encoder(x)

            # 解码器前向传播
            x = self.decoder(features)

            # 上采样到原始尺寸
            x = F.interpolate(
                x, size=(512, 512), mode='bilinear', align_corners=False
            )

            return x

    # 创建解码器
    decoder = DeepLabHead(encoder_channels, num_classes)

    # 创建完整模型
    model = DeepLabv3Plus(encoder, decoder, num_classes)

    return model