# PyTorch EfficientPose for Pallet Detection

## 安装

```bash
pip install -r requirements.txt
```

## 训练

```bash
python train.py
```

## 推理

```bash
python inference.py
```

## 项目结构

```
pytorch_efficientpose/
├── config.py              # 配置文件
├── train.py               # 训练脚本
├── inference.py           # 推理脚本
├── models/
│   ├── efficientnet.py    # Backbone
│   ├── bifpn.py           # 特征金字塔
│   └── efficientpose.py   # 主模型
├── datasets/
│   └── pallet_dataset.py  # 数据加载
├── losses/
│   └── pose_loss.py       # 损失函数
└── requirements.txt
```
