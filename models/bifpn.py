"""
BiFPN (Bidirectional Feature Pyramid Network)
用于多尺度特征融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    """深度可分离卷积"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class BiFPNLayer(nn.Module):
    """单层 BiFPN"""
    def __init__(self, num_channels, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon

        # 自上而下路径的权重 (learnable)
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32))

        # 卷积层
        self.conv_p3_td = SeparableConv2d(num_channels, num_channels)
        self.conv_p4_td = SeparableConv2d(num_channels, num_channels)

        self.conv_p4_out = SeparableConv2d(num_channels, num_channels)
        self.conv_p5_out = SeparableConv2d(num_channels, num_channels)

    def forward(self, features):
        """
        Args:
            features: [P3, P4, P5]
        Returns:
            outputs: [P3_out, P4_out, P5_out]
        """
        p3, p4, p5 = features

        # 自上而下路径
        # P4_td = w1[0]*P4 + w1[1]*Upsample(P5)
        w1 = F.relu(self.w1)
        w1 = w1 / (torch.sum(w1, dim=0) + self.epsilon)
        p4_td = self.conv_p4_td(
            w1[0] * p4 + w1[1] * F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        )

        # P3_td = w1[0]*P3 + w1[1]*Upsample(P4_td)
        p3_td = self.conv_p3_td(
            w1[0] * p3 + w1[1] * F.interpolate(p4_td, size=p3.shape[-2:], mode='nearest')
        )

        # 自下而上路径
        w2 = F.relu(self.w2)
        w2 = w2 / (torch.sum(w2, dim=0) + self.epsilon)

        # P4_out = w2[0]*P4 + w2[1]*P4_td + w2[2]*Downsample(P3_td)
        p3_down = F.max_pool2d(p3_td, kernel_size=2, stride=2)
        p4_out = self.conv_p4_out(
            w2[0] * p4 + w2[1] * p4_td + w2[2] * p3_down
        )

        # P5_out = w1[0]*P5 + w1[1]*Downsample(P4_out)
        p4_down = F.max_pool2d(p4_out, kernel_size=2, stride=2)
        p5_out = self.conv_p5_out(
            w1[0] * p5 + w1[1] * p4_down
        )

        return [p3_td, p4_out, p5_out]

class BiFPN(nn.Module):
    """多层 BiFPN"""
    def __init__(self, in_channels_list, num_channels, num_layers=3):
        super().__init__()

        # 输入投影层（将不同通道数统一为 num_channels）
        self.input_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, num_channels, kernel_size=1),
                nn.BatchNorm2d(num_channels)
            ) for in_ch in in_channels_list
        ])

        # BiFPN 层
        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(num_channels) for _ in range(num_layers)
        ])

    def forward(self, features):
        """
        Args:
            features: [P3, P4, P5] from backbone
        Returns:
            outputs: [P3_out, P4_out, P5_out]
        """
        # 投影到统一通道数
        features = [conv(feat) for conv, feat in zip(self.input_convs, features)]

        # 多层 BiFPN
        for bifpn_layer in self.bifpn_layers:
            features = bifpn_layer(features)

        return features

if __name__ == "__main__":
    # 测试
    in_channels = [40, 112, 320]  # EfficientNet-B0 的输出通道
    bifpn = BiFPN(in_channels, num_channels=64, num_layers=3)

    features = [
        torch.randn(2, 40, 64, 64),   # P3
        torch.randn(2, 112, 32, 32),  # P4
        torch.randn(2, 320, 16, 16),  # P5
    ]

    outputs = bifpn(features)
    print("BiFPN 测试:")
    for i, out in enumerate(outputs):
        print(f"  P{i+3}_out: {out.shape}")
