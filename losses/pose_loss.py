"""
位姿估计损失函数
包含分类损失、bbox回归损失、旋转损失、平移损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for classification"""
    def __init__(self, alpha=0.75, gamma=1.75):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma


    def forward(self, pred, target):
        """
        pred:   [B, N, 1]  模型原始输出（已经过 Sigmoid，值在 [0,1]）
        target: [B, N, 1]  0 或 1
        """
        pred = pred.float()
        target = target.float()

        # ★ 核心修改：不用 binary_cross_entropy，改用数值稳定版本
        # 先把 sigmoid 概率反推回 logit，再用 with_logits 版本
        # 等价于：彻底用 logit 流，但兼容模型已有 Sigmoid 输出
        pred_clamped = torch.clamp(pred, min=1e-6, max=1.0 - 1e-6)
        logit = torch.log(pred_clamped / (1.0 - pred_clamped))  # 反 sigmoid

        bce_loss = F.binary_cross_entropy_with_logits(
            logit, target, reduction='none'
        )

        pt = torch.where(target == 1, pred_clamped, 1.0 - pred_clamped)
        focal_weight = self.alpha * (1.0 - pt) ** self.gamma

        return (focal_weight * bce_loss).mean()




class SmoothL1Loss(nn.Module):
    """Smooth L1 Loss for bbox regression"""
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: [B, N, 4] predicted bboxes
            target: [B, N, 4] target bboxes
            mask: [B, N] valid object mask
        """
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )

        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            return loss.sum() / (mask.sum() + 1e-6)
        else:
            return loss.mean()



class RotationLoss(nn.Module):
    def forward(self, pred_rot6d, target_rot, pos_mask):
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_rot6d.device, requires_grad=True)

        B = pred_rot6d.shape[0]

        # ★ 兼容 [B, 9] 和 [B, N, 9] 两种形状
        if target_rot.dim() == 3:
            # [B, N, 9] → 取第一个 anchor 的 gt（所有 anchor gt 相同）
            target_rot = target_rot[:, 0, :]   # [B, 9]

        # gt 旋转矩阵：[B, 3, 3]
        gt_R = target_rot.reshape(B, 3, 3)

        # 取前两列转成 6D：[B, 6]
        gt_6d = torch.cat([gt_R[..., 0], gt_R[..., 1]], dim=-1)  # [B, 6]

        # 扩展到每个 anchor：[B, N, 6]
        gt_6d_expanded = gt_6d.unsqueeze(1).expand_as(pred_rot6d)

        mask = pos_mask.unsqueeze(-1)
        loss = F.smooth_l1_loss(
            pred_rot6d * mask,
            gt_6d_expanded * mask,
            reduction='sum'
        ) / (pos_mask.sum() + 1e-6)

        return loss





class TranslationLoss(nn.Module):
    """平移向量损失（Euclidean distance）"""
    def __init__(self):
        super().__init__()

    def forward(self, pred_t, target_t, mask=None):
        """
        Args:
            pred_t: [B, N, 3] predicted translation
            target_t: [B, N, 3] target translation
            mask: [B, N] valid object mask
        """

        if torch.isnan(pred_t).any() or torch.isinf(pred_t).any():
            return torch.tensor(0.0, device=pred_t.device, requires_grad=True)
        

        diff = pred_t - target_t
        loss_per_sample = torch.norm(diff, dim=-1)  # [B, N]

        if mask is not None:
            loss = (loss_per_sample * mask).sum() / (mask.sum() + 1e-6)
        else:
            loss = loss_per_sample.mean()

        return loss


class TransformationLoss(nn.Module):
    """
    变换损失：测量变换后3D点的距离
    基于 EfficientPose 论文的 Transformation Loss
    """
    def __init__(self, model_points):
        """
        Args:
            model_points: [num_points, 3] 托盘3D模型点（如8个角点）
        """
        super().__init__()
        
        # 强制转换为 torch.Tensor
        if not isinstance(model_points, torch.Tensor):
            model_points = torch.tensor(model_points, dtype=torch.float32)
        
        # 注册为 buffer，会自动跟随模型移动到 GPU/CPU
        self.register_buffer('model_points', model_points.float())
        self.num_points = model_points.shape[0]

    def forward(self, pred_rot, pred_trans, target_rot, target_trans, pos_mask, model_points):
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_rot.device, requires_grad=True)

        B, N, _ = pred_rot.shape   # [B, N, 6]

        # ★ 兼容 [B, N, 9] 旧格式 和 [B, N, 6] 新格式
        if pred_rot.shape[-1] == 6:
            # 6D → 旋转矩阵
            pred_R = rot6d_to_matrix(pred_rot)              # [B, N, 3, 3]
            pred_R = pred_R.reshape(B * N, 3, 3)
        else:
            # 旧的 9D SVD 方式（保留兼容）
            pred_R = pred_rot.reshape(B * N, 3, 3)

        # gt 旋转矩阵
        if target_rot.dim() == 3:
            target_rot = target_rot[:, 0, :]   # [B, 9]
        gt_R = target_rot.reshape(B, 3, 3)
        gt_R = gt_R.unsqueeze(1).expand(B, N, 3, 3).reshape(B * N, 3, 3)

        # 平移
        if target_trans.dim() == 3:
            target_trans = target_trans[:, 0, :]   # [B, 3]
        gt_t = target_trans.unsqueeze(1).expand(B, N, 3).reshape(B * N, 3)
        pred_t = pred_trans.reshape(B * N, 3)

        # model_points: [M, 3]
        M = model_points.shape[0]
        pts = model_points.to(pred_rot.device)              # ★ 移到同一设备
        pts = pts.unsqueeze(0).expand(B * N, -1, -1)        # [B*N, M, 3]

        pred_pts = (pred_R @ pts.transpose(1, 2)).transpose(1, 2) + pred_t.unsqueeze(1)
        gt_pts   = (gt_R   @ pts.transpose(1, 2)).transpose(1, 2) + gt_t.unsqueeze(1)

        dist = torch.norm(pred_pts - gt_pts, dim=-1).mean(dim=-1)   # [B*N]

        mask_flat = pos_mask.reshape(B * N)
        loss = (dist * mask_flat).sum() / (mask_flat.sum() + 1e-6)
        return loss



def compute_iou(bbox1, bbox2):
    """
    计算两个 bbox 的 IoU
    Args:
        bbox1: [B, N, 4] (x1, y1, x2, y2) 归一化坐标 [0, 1]
        bbox2: [B, 4] (x1, y1, x2, y2) 归一化坐标 [0, 1]
    Returns:
        iou: [B, N] IoU 值
    """
    B, N = bbox1.shape[:2]
    
    # 扩展 bbox2 到 [B, N, 4]
    bbox2 = bbox2.unsqueeze(1).expand(-1, N, -1)
    
    # 计算交集
    x1_max = torch.max(bbox1[..., 0], bbox2[..., 0])
    y1_max = torch.max(bbox1[..., 1], bbox2[..., 1])
    x2_min = torch.min(bbox1[..., 2], bbox2[..., 2])
    y2_min = torch.min(bbox1[..., 3], bbox2[..., 3])
    
    inter_w = torch.clamp(x2_min - x1_max, min=0)
    inter_h = torch.clamp(y2_min - y1_max, min=0)
    inter_area = inter_w * inter_h
    
    # 计算并集
    area1 = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
    area2 = (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1])
    union_area = area1 + area2 - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-6)
    return iou

def generate_anchors(image_size=512, feature_size=64, num_anchors=9):
    """
    生成固定的 anchor 先验框（归一化坐标）
    覆盖不同尺度和长宽比
    """
    scales = [0.5, 1.0, 2.0]          # 3种尺度
    ratios = [0.5, 1.0, 2.0]          # 3种长宽比
    stride = 1.0 / feature_size       # 每个格子的步长

    anchors = []
    for i in range(feature_size):
        for j in range(feature_size):
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride
            for s in scales:
                for r in ratios:
                    w = stride * s * (r ** 0.5)
                    h = stride * s / (r ** 0.5)
                    anchors.append([cx - w/2, cy - h/2,
                                    cx + w/2, cy + h/2])
    return torch.tensor(anchors, dtype=torch.float32)  # [N, 4]

def assign_anchors_to_gt(predictions, targets,
                          image_size=512, feature_size=64,
                          num_anchors=9, iou_threshold=0.5, 
                          anchors=None):
    B = predictions['bbox'].shape[0]
    N = predictions['bbox'].shape[1]
    device = predictions['bbox'].device

    if anchors is None:
        anchors = generate_anchors(image_size, feature_size, num_anchors)
    anchors_expanded = anchors.to(device).unsqueeze(0).expand(B, -1, -1)

    gt_bbox = targets['bbox']
    iou = compute_iou(anchors_expanded, gt_bbox)

    mask = (iou > iou_threshold).float()

    # ★ 每张图取 Top-K 个正样本，而不是只取 1 个
    K = 10  # 至少保证 10 个正样本
    topk_idx = torch.topk(iou, k=min(K, N), dim=1).indices  # [B, K]
    for b in range(B):
        mask[b, topk_idx[b]] = 1.0

    neg_mask = (iou < 0.2).float()  # ★ 0.4→0.2，避免正负样本重叠

    return mask, neg_mask


def rot6d_to_matrix(rot6d):
    """
    6D 旋转表示 → 3×3 旋转矩阵
    rot6d: [..., 6]  →  R: [..., 3, 3]
    """
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)   # [..., 3, 3]


class PoseLoss(nn.Module):
    """组合损失函数"""
    def __init__(self, config):
        super().__init__()
        self.focal_loss = FocalLoss()
        self.smooth_l1_loss = SmoothL1Loss()
        self.rotation_loss = RotationLoss()
        self.translation_loss = TranslationLoss()
        
        # 获取 3D 模型点（托盘的 8 个角点）
        model_points = config.get_model_points()  # 应该返回 [8, 3] 的 tensor
        self.transformation_loss = TransformationLoss(model_points)

        self.register_buffer('model_points', config.get_model_points())

        self.weights = config.loss_weights
        
        # 🔥 新增：anchor 匹配参数
        self.image_size = config.image_size
        self.feature_size = self.image_size // 8  # P3 层降采样 8 倍
        self.num_anchors = 9
        self.register_buffer(
            'anchors',
            generate_anchors(self.image_size, self.feature_size, self.num_anchors)
        )

    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict from model
                - 'bbox': [B, N, 4]
                - 'class': [B, N, 1]
                - 'rotation': [B, N, 9]
                - 'translation': [B, N, 3]
            targets: dict from dataloader
                - 'bbox': [B, 4]
                - 'rotation': [B, 3, 3]
                - 'translation': [B, 3]
        Returns:
            dict of losses
        """
        B = predictions['bbox'].shape[0]
        N = predictions['bbox'].shape[1]
        device = predictions['bbox'].device

        # 🔥 基于 IoU 匹配 anchor
        pos_mask, neg_mask = assign_anchors_to_gt(
            predictions, targets,
            image_size=self.image_size,
            feature_size=self.feature_size,
            num_anchors=self.num_anchors,
            iou_threshold=0.5,
            anchors=self.anchors        # ← 传入预生成的 anchor
        )

        
        # 扩展 targets 到 [B, N, *]
        target_bbox = targets['bbox'].unsqueeze(1).expand(-1, N, -1)
        
        # 🔥 分类目标：正样本=1，负样本=0
        target_class = pos_mask.unsqueeze(-1)  # [B, N, 1]
        
        # 旋转矩阵展平: [B, 3, 3] -> [B, 9] -> [B, N, 9]
        target_rotation = targets['rotation'].reshape(B, 9).unsqueeze(1).expand(-1, N, -1)
        target_translation = targets['translation'].unsqueeze(1).expand(-1, N, -1)

        # 🔥 修复：分类损失使用 Focal Loss
        cls_loss = self.focal_loss(predictions['class'], target_class)
        
        # 🔥 回归损失：只在正样本上计算
        bbox_loss = self.smooth_l1_loss(predictions['bbox'], target_bbox, pos_mask)
        rot_loss = self.rotation_loss(predictions['rotation'], target_rotation, pos_mask)
        trans_loss = self.translation_loss(predictions['translation'], target_translation, pos_mask)

        # Transformation loss
        transform_loss = self.transformation_loss(
            predictions['rotation'], 
            predictions['translation'],
            target_rotation, 
            target_translation,
            pos_mask,
            self.model_points
        )

        # 加权组合
        total_loss = (
            self.weights['classification'] * cls_loss +
            self.weights['bbox'] * bbox_loss +
            self.weights['rotation'] * rot_loss +
            self.weights['translation'] * trans_loss +
            self.weights['transformation'] * transform_loss
        )

        return {
            'total': total_loss,
            'classification': cls_loss,
            'bbox': bbox_loss,
            'rotation': rot_loss,
            'translation': trans_loss,
            'transformation': transform_loss
        }



if __name__ == "__main__":
    from config import Config
    config = Config()

    criterion = PoseLoss(config)

    # 模拟预测和目标
    B, N = 2, 100
    predictions = {
        'bbox': torch.rand(B, N, 4),  # 归一化坐标 [0, 1]
        'class': torch.sigmoid(torch.randn(B, N, 1)),
        'rotation': torch.randn(B, N, 9),
        'translation': torch.randn(B, N, 3)
    }

    targets = {
        'bbox': torch.rand(B, 4),  # 归一化坐标 [0, 1]
        'rotation': torch.eye(3).unsqueeze(0).expand(B, -1, -1),
        'translation': torch.randn(B, 3)
    }

    losses = criterion(predictions, targets)
    print("损失测试:")
    for key, val in losses.items():
        print(f"  {key}: {val.item():.4f}")
