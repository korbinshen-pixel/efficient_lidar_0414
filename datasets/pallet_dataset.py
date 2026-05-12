"""
Pallet Dataset for EfficientPose Training
支持 RGB + LiDAR 点云融合（将点云投影为伪深度图后拼接）
"""
import os
import cv2
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from config import Config


def project_lidar_to_image(points_cam: np.ndarray,
                            K: np.ndarray,
                            img_h: int, img_w: int,
                            max_depth: float = 10.0) -> np.ndarray:
    """
    将相机光学坐标系下的 LiDAR 点云投影为伪深度图

    Args:
        points_cam: (N, 3) float32，相机光学坐标系，单位 m
                    （dataset_collector.py 已经做了坐标变换，直接用）
        K:          (3, 3) 相机内参矩阵
        img_h, img_w: 图像尺寸
        max_depth:  最大有效深度（m），超出的点丢弃
    Returns:
        depth_map: (H, W) float32，归一化到 [0, 1]
    """
    depth_map = np.zeros((img_h, img_w), dtype=np.float32)

    if len(points_cam) == 0:
        return depth_map

    # 只保留在相机前方的点
    mask = points_cam[:, 2] > 0.1
    pts = points_cam[mask]

    if len(pts) == 0:
        return depth_map

    # 投影到像素坐标
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = pts[:, 2]
    u = (fx * pts[:, 0] / z + cx).astype(np.int32)
    v = (fy * pts[:, 1] / z + cy).astype(np.int32)

    # 过滤出图像范围内的点
    valid = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h) & (z < max_depth)
    u, v, z = u[valid], v[valid], z[valid]

    # 对同一像素取最近点（z-buffer）
    # 从远到近写入，保证近点覆盖远点
    order = np.argsort(z)[::-1]
    depth_map[v[order], u[order]] = z[order]

    # 归一化到 [0, 1]
    depth_map = np.clip(depth_map / max_depth, 0.0, 1.0)

    return depth_map


class PalletDataset(Dataset):
    """托盘数据集 - Linemod 格式，支持 RGB + LiDAR 点云融合"""

    def __init__(self, dataset_path, object_dir='01', split='train',
                 transform=None, config=None):
        self.dataset_path = Path(dataset_path).expanduser()
        self.data_dir     = self.dataset_path / object_dir
        self.split        = split
        self.transform    = transform
        self.config       = config

        print(f"正在加载数据集: {self.data_dir}")

        # ── 读取分割文件 ──
        split_file = self.data_dir / f'{split}.txt'
        if not split_file.exists():
            raise FileNotFoundError(f"找不到 {split_file}")
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f if line.strip()]

        # ── 读取标注文件 ──
        with open(self.data_dir / 'gt.yml', 'r') as f:
            self.gt_dict = yaml.safe_load(f)
        with open(self.data_dir / 'info.yml', 'r') as f:
            self.info_dict = yaml.safe_load(f)

        # ── ★ 检测 LiDAR 目录（替换原来的 depth 目录检测）──
        self.lidar_dir  = self.data_dir / 'lidar'
        self.depth_dir  = self.data_dir / 'depth'   # 保留兼容，但不用于训练

        self.use_lidar = self.lidar_dir.exists()
        self.use_depth = False   # ★ 明确禁用深度图

        if self.use_lidar:
            print(f"✅ 检测到 LiDAR 点云目录，将使用 RGB + LiDAR 4通道输入")
        else:
            print(f"⚠️  未找到 LiDAR 目录 ({self.lidar_dir})，使用纯 RGB 3通道输入")

        print(f"✅ 加载 {split} 数据集: {len(self.image_ids)} 帧")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        frame_id = int(image_id)

        # ── 读取 RGB ──
        rgb_path = self.data_dir / 'rgb' / f'{image_id}.png'
        image = cv2.imread(str(rgb_path))
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {rgb_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w  = image.shape[:2]

        # ── 读取相机内参（用于点云投影）──
        if frame_id not in self.info_dict:
            raise KeyError(f"info.yml 中找不到帧 {frame_id}")
        info_data = self.info_dict[frame_id]
        info = info_data[0] if isinstance(info_data, list) else info_data
        K = np.array(info['cam_K'], dtype=np.float32).reshape(3, 3)


        # ── 读取 mask ──
        mask_path = self.data_dir / 'mask' / f'{image_id}.png'
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = np.ones((h, w), dtype=np.uint8)

        # 统一 resize 到模型输入尺寸
        target_size = Config.image_size if Config else 512  # 512

        image = cv2.resize(image, (target_size, target_size))
        mask  = cv2.resize(mask,  (target_size, target_size), interpolation=cv2.INTER_NEAREST)  # mask 用最近邻，不产生中间值


        # bbox 不需要改，因为本来就是归一化坐标 [0,1]，与分辨率无关

        # ── ★ 读取 LiDAR 点云并投影为伪深度图 ──
        if self.use_lidar:
            lidar_path = self.lidar_dir / f'{image_id}.npy'
            if lidar_path.exists():
                pts_cam = np.load(str(lidar_path))   # (N, 3)
            else:
                pts_cam = np.zeros((0, 3), dtype=np.float32)

            lidar_depth = project_lidar_to_image(pts_cam, K, h, w, max_depth=10.0)
            # ★ resize 到和 image 一致
            lidar_depth = cv2.resize(lidar_depth, (target_size, target_size),
                                     interpolation=cv2.INTER_LINEAR)

        # ── 读取位姿标注 ──
        if frame_id not in self.gt_dict:
            raise KeyError(f"gt.yml 中找不到帧 {frame_id}")
        gt_data = self.gt_dict[frame_id]
        gt = gt_data[0] if isinstance(gt_data, list) else gt_data

        R = np.array(gt['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
        t = np.array(gt['cam_t_m2c'], dtype=np.float32) / 1000.0  # mm → m

        # ── 从 mask 计算 2D bbox（归一化）──
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
        else:
            bbox = np.array([0, 0, w - 1, h - 1], dtype=np.float32)
        bbox = bbox / np.array([w, h, w, h], dtype=np.float32)

        # ── 数据增强（仅作用于 RGB + mask，不影响点云）──
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask  = augmented['mask']

        # ── ★ 转 Tensor：RGB 归一化 + LiDAR 伪深度图拼接 ──
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_t = (image_t - mean) / std    # [3, H, W]

        if self.use_lidar:
            lidar_t = torch.from_numpy(lidar_depth).unsqueeze(0)  # [1, H, W]
            image_t = torch.cat([image_t, lidar_t], dim=0)        # [4, H, W]

        return {
            'image':          image_t,
            'bbox':           torch.from_numpy(bbox),
            'rotation':       torch.from_numpy(R),
            'translation':    torch.from_numpy(t),
            'camera_matrix':  torch.from_numpy(K),
            'mask':           torch.from_numpy(mask.astype(np.float32)),
            'image_id':       image_id
        }


def collate_fn(batch):
    return {
        'image':          torch.stack([b['image']         for b in batch]),
        'bbox':           torch.stack([b['bbox']          for b in batch]),
        'rotation':       torch.stack([b['rotation']      for b in batch]),
        'translation':    torch.stack([b['translation']   for b in batch]),
        'camera_matrix':  torch.stack([b['camera_matrix'] for b in batch]),
        'mask':           torch.stack([b['mask']          for b in batch]),
        'image_ids':      [b['image_id'] for b in batch]
    }


