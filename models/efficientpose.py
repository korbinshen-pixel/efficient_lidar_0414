"""
EfficientPose - 完整的 6DoF 位姿估计模型
"""
import torch
import torch.nn as nn
from .efficientnet import EfficientNetBackbone
from .bifpn import BiFPN

class RegressionHead(nn.Module):
    """2D bbox 回归头"""
    def __init__(self, in_channels, num_anchors=9):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)

class ClassificationHead(nn.Module):
    """分类头"""
    def __init__(self, in_channels, num_classes, num_anchors=9):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)

class RotationHead(nn.Module):
    def __init__(self, in_channels, num_anchors=9):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * 6, kernel_size=3, padding=1),  # ★ 9→6
            nn.Tanh()   # ★ 限制值域，防止爆炸
        )

    def forward(self, x):
        return self.conv(x)


class TranslationHead(nn.Module):
    """平移向量回归头（输出 tx, ty, tz）"""
    def __init__(self, in_channels, num_anchors=9):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.conv(x)

class EfficientPose(nn.Module):
    """EfficientPose 完整模型"""

    def __init__(self, phi=0, num_classes=1, pretrained=True, in_channels=3):
        super().__init__()
        self.phi = phi
        self.num_classes = num_classes

        # Backbone
        self.backbone = EfficientNetBackbone(phi=phi, pretrained=pretrained, in_channels=in_channels)

        # 获取 backbone 输出通道数
        in_channels_list = self.backbone.feature_info

        # BiFPN 特征金字塔
        self.bifpn_channels = 64 + phi * 16  # 动态调整通道数
        self.bifpn = BiFPN(
            in_channels_list=in_channels_list,
            num_channels=self.bifpn_channels,
            num_layers=3 + phi
        )

        # 预测头
        self.num_anchors = 9
        self.regression_head = RegressionHead(self.bifpn_channels, self.num_anchors)
        self.classification_head = ClassificationHead(self.bifpn_channels, num_classes, self.num_anchors)
        self.rotation_head = RotationHead(self.bifpn_channels, self.num_anchors)
        self.translation_head = TranslationHead(self.bifpn_channels, self.num_anchors)

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            dict with keys:
                - 'bbox': [B, N, 4] - 2D bbox (x1, y1, x2, y2)
                - 'class': [B, N, num_classes] - 分类概率
                - 'rotation': [B, N, 9] - 旋转矩阵展平
                - 'translation': [B, N, 3] - 平移向量 (tx, ty, tz)
        """
        # Backbone 提取特征
        features = self.backbone(x)  # [P3, P4, P5]

        # BiFPN 融合特征
        features = self.bifpn(features)  # [P3', P4', P5']

        # 对每个特征层应用预测头（这里简化为只用 P3）
        feat = features[0]  # 使用 P3 特征

        # 预测
        bbox_pred = self.regression_head(feat)  # [B, 36, H, W]
        class_pred = self.classification_head(feat)  # [B, 9, H, W]
        rotation_pred = self.rotation_head(feat)  # [B, 81, H, W]
        translation_pred = self.translation_head(feat)  # [B, 27, H, W]

        # Reshape 为 [B, N, *]
        B = x.shape[0]
        H, W = feat.shape[-2:]
        N = self.num_anchors * H * W

        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(B, N, 4)
        class_pred = class_pred.permute(0, 2, 3, 1).reshape(B, N, self.num_classes)
        rotation_pred = rotation_pred.permute(0, 2, 3, 1).reshape(B, N, 6)
        translation_pred = translation_pred.permute(0, 2, 3, 1).reshape(B, N, 3)

        return {
            'bbox': bbox_pred,
            'class': class_pred,
            'rotation': rotation_pred,
            'translation': translation_pred
        }

    def freeze_backbone(self):
        """冻结 backbone 权重"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """解冻 backbone 权重"""
        for param in self.backbone.parameters():
            param.requires_grad = True

if __name__ == "__main__":
    # 测试
    model = EfficientPose(phi=0, num_classes=1, pretrained=False)
    x = torch.randn(2, 3, 512, 512)

    outputs = model(x)

    print("EfficientPose 测试:")
    for key, val in outputs.items():
        print(f"  {key}: {val.shape}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
