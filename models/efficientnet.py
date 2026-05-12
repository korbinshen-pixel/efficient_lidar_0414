"""
EfficientNet Backbone - 使用 timm 库，支持本地预训练权重
"""
import os
import glob
import torch
import torch.nn as nn
import timm


def _load_weights(path):
    """支持 .safetensors 和 .pth/.bin（模块级函数，避免 nn.Module.__getattr__ 冲突）"""
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(path)
    else:
        return torch.load(path, map_location='cpu')


class EfficientNetBackbone(nn.Module):

    def __init__(self, phi=0, pretrained=True, in_channels=3):
        super().__init__()
        self.phi = phi

        model_names = {
            0: 'efficientnet_b0', 1: 'efficientnet_b1',
            2: 'efficientnet_b2', 3: 'efficientnet_b3',
            4: 'efficientnet_b4', 5: 'efficientnet_b5',
            6: 'efficientnet_b6',
        }
        name = model_names[phi]

        # ★ 先创建空模型，不触发网络下载
        self.model = timm.create_model(
            name,
            pretrained=False,
            features_only=True,
            out_indices=(2, 3, 4)
        )
        self.feature_info = self.model.feature_info.channels()

        # ★ 从本地加载预训练权重
        if pretrained:
            local_patterns = [
                # HuggingFace 缓存路径（自动匹配所有 snapshot）
                os.path.expanduser(
                    f'~/.cache/huggingface/hub/models--timm--{name}.ra_in1k'
                    f'/snapshots/**/model.safetensors'
                ),
                # 手动放在工作目录的权重
                f'/home/usr2/data_trainer_ws/efficinet20260307/{name}.safetensors',
                f'/home/usr2/data_trainer_ws/efficinet20260307/{name}.pth',
            ]
            loaded = False
            for pattern in local_patterns:
                matches = glob.glob(pattern, recursive=True)
                if not matches:
                    continue
                p = matches[0]
                print(f"✅ 从本地加载预训练权重: {p}")
                state = _load_weights(p)
                missing, unexpected = self.model.load_state_dict(state, strict=False)
                print(f"   missing: {len(missing)}, unexpected: {len(unexpected)}")
                loaded = True
                break
            if not loaded:
                print("⚠️  未找到本地预训练权重，从零开始训练")

        # ★ 深度通道融合
        self.use_depth = (in_channels == 4)
        if self.use_depth:
            self.depth_proj = nn.Conv2d(1, 3, kernel_size=1, bias=False)
            nn.init.constant_(self.depth_proj.weight, 1.0 / 3.0)
            print("✅ EfficientNetBackbone: 已启用深度通道融合 (RGBD → RGB)")

    def forward(self, x):
        if self.use_depth and x.shape[1] == 4:
            rgb   = x[:, :3]
            depth = x[:, 3:4]
            x = rgb + self.depth_proj(depth)
        return self.model(x)


if __name__ == '__main__':
    for in_ch in [3, 4]:
        model = EfficientNetBackbone(phi=0, pretrained=True, in_channels=in_ch)
        x = torch.randn(2, in_ch, 512, 512)
        feats = model(x)
        print(f"in_channels={in_ch}:")
        for i, f in enumerate(feats):
            print(f"  P{i+3}: {f.shape}")
