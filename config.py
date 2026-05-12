"""
配置文件 - PyTorch EfficientPose for Pallet Detection
"""
import torch

class Config:
    # ========== 模型参数 ==========
    phi = 0  # 模型规模 [0-6]，0最小最快，6最大最精确
    num_classes = 1  # 托盘只有1个类别

    # Compound scaling coefficients for different phi values
    compound_coef = {
        0: {'width': 1.0, 'depth': 1.0, 'resolution': 512},
        1: {'width': 1.0, 'depth': 1.1, 'resolution': 640},
        2: {'width': 1.1, 'depth': 1.2, 'resolution': 768},
        3: {'width': 1.2, 'depth': 1.4, 'resolution': 896},
        4: {'width': 1.4, 'depth': 1.8, 'resolution': 1024},
        5: {'width': 1.6, 'depth': 2.2, 'resolution': 1280},
        6: {'width': 1.8, 'depth': 2.6, 'resolution': 1280},
    }

    # ========== 数据集参数 ==========
    dataset_path = '/home/usr2/data_trainer_ws/efficinet20260307'
    object_dir = '20260412_155551_01'
    image_size = compound_coef[phi]['resolution']

    # 托盘3D尺寸 (米)
    pallet_dimensions = {
        'length': 0.24,   # x
        'width':  0.20,   # y
        'height': 0.035  # z（总高度）
    }

    # 相机内参 (需根据实际相机修改)
    camera_matrix = torch.tensor([
        [519.25, 0,     320.0],
        [0,     519.25, 240.0],
        [0,     0,     1.0  ]
    ], dtype=torch.float32)

    # ========== 训练参数 ==========
    batch_size = 16
    num_epochs = 400
    learning_rate = 1e-4
    weight_decay = 1e-5

    # 学习率调度
    lr_scheduler = 'cosine'  # 'step', 'cosine', 'plateau'
    lr_step_size = 50
    lr_gamma = 0.1

    # Early stopping
    patience = 100

    # pretrained = False   # 先跑通流程，之后再换预训练权重

    # ========== 损失权重 ==========
    loss_weights = {
        'classification': 100.0,
        'bbox': 50.0,
        'rotation': 2.0,      # 旋转损失权重
        'translation': 5.0,   # 平移损失权重
        'transformation': 1.0  # 整体变换损失
    }

    # ========== 数据增强 ==========
    augmentation = {
        'horizontal_flip': 0.5,
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1,
        'noise_std': 0.01,
    }

    # ========== 优化器参数 ==========
    optimizer = 'adamw'  # 'adam', 'adamw', 'sgd'
    momentum = 0.9  # for SGD

    # ========== 硬件参数 ==========
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4
    pin_memory = True

    # ========== 输出路径 ==========
    checkpoints_dir = './checkpoints'
    logs_dir = './logs'
    results_dir = './results'

    # ========== 推理参数 ==========
    confidence_threshold = 0.5
    nms_threshold = 0.3
    max_detections = 10

    # ========== 8个关键点定义（托盘3D bbox的8个角点）==========
    # 相对于托盘中心的坐标
    @staticmethod
    def get_model_points():
        """返回托盘3D bbox的8个角点"""
        l, w, h = Config.pallet_dimensions['length'], Config.pallet_dimensions['width'], Config.pallet_dimensions['height']
        return torch.tensor([
            [-l/2, -w/2, 0],  # 0: 左下后
            [ l/2, -w/2, 0],  # 1: 右下后
            [ l/2,  w/2, 0],  # 2: 右上后
            [-l/2,  w/2, 0],  # 3: 左上后
            [-l/2, -w/2,  h],  # 4: 左下前
            [ l/2, -w/2,  h],  # 5: 右下前
            [ l/2,  w/2,  h],  # 6: 右上前
            [-l/2,  w/2,  h],  # 7: 左上前
        ], dtype=torch.float32)

def get_config(phi=0):
    """获取指定 phi 值的配置"""
    config = Config()
    config.phi = phi
    config.image_size = config.compound_coef[phi]['resolution']
    return config
