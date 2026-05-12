"""
EfficientPose 模型评估脚本（含图像可视化输出）
评估指标：
- Detection Rate: 检测成功率
- ADD: Average Distance of Model Points (非对称物体)
- ADD-S: ADD-Symmetric (对称物体)
- 2D IoU: 2D 边界框的 IoU
- Rotation Error: 旋转误差 (度)
- Translation Error: 平移误差 (cm)

可视化输出（保存到 results/ 目录）：
- per_sample/: 每张测试图的预测结果叠加图
- eval_grid.jpg: 最多 16 张结果拼图
- eval_metrics.png: ADD / 旋转误差 / 平移误差折线图
- eval_add_curve.png: ADD AUC 曲线
- eval_trans_hist.png: 平移误差分布直方图（忽略 > 20cm）
- eval_rot_hist.png: 旋转误差分布直方图（忽略 > 10°）
"""
import os
import sys
import cv2
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from config import Config
from models.efficientpose import EfficientPose
from datasets.pallet_dataset import PalletDataset


# ─────────────────────────────────────────────────────────────
#  可视化工具函数
# ─────────────────────────────────────────────────────────────

# def denormalize_image(tensor):
#     """将归一化 tensor 还原为 uint8 numpy BGR 图像（兼容 RGB 和 RGBD）"""
#     mean = np.array([0.485, 0.456, 0.406])
#     std  = np.array([0.229, 0.224, 0.225])
#     img = tensor.cpu().permute(1, 2, 0).numpy()  # [H, W, C]
    
#     # ★ 只取前 3 通道（RGB），深度通道丢弃不显示
#     img = img[:, :, :3]
    
#     img = img * std + mean
#     img = np.clip(img * 255, 0, 255).astype(np.uint8)
#     return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def denormalize_image(tensor):
    """将归一化 tensor 还原为 uint8 numpy BGR 图像（兼容 RGB 和 RGBD/LiDAR）"""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().permute(1, 2, 0).numpy()  # [H, W, C]

    # ★ 只取前 3 通道（RGB），第4通道（LiDAR伪深度图）丢弃不显示
    img = img[:, :, :3]

    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)




def draw_bbox(img, bbox_norm, color, label='', thickness=2,
              scale=1.0, y_shift=0.0):
    """
    对归一化 bbox 做简单几何补偿：
    - scale > 1: 以中心为基准放大
    - y_shift < 0: 整体向上平移（单位：相对图像高度）
    只影响可视化，评估指标仍用原始 bbox 计算。
    """
    import numpy as np

    h, w = 480, 640
    x1, y1, x2, y2 = map(float, bbox_norm)

    # 中心点
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # 放大宽高
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale

    # 向上平移（减 y_shift）
    cy = cy + y_shift

    x1 = (cx - bw / 2.0) * w
    x2 = (cx + bw / 2.0) * w
    y1 = (cy - bh / 2.0) * h
    y2 = (cy + bh / 2.0) * h

    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, 0, h - 1))

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img



def project_3d_axes(img, R, t, K, axis_len=0.05, y_shift_rel=0.0):
    """
    将 3D 坐标轴投影到图像上，用于直观展示姿态
    X 轴 = 红色，Y 轴 = 绿色，Z 轴 = 蓝色
    y_shift_rel: 相对于图像高度的整体上下偏移（负值=向上）
    """
    h, w = img.shape[:2]

    origin = np.array([
        [0, 0, 0],
        [axis_len, 0, 0],   # X
        [0, axis_len, 0],   # Y
        [0, 0, axis_len],   # Z
    ], dtype=np.float32)  # [4,3]

    # 世界到相机
    pts_cam = (R @ origin.T + t.reshape(3, 1)).T      # [4,3]
    proj = (K @ pts_cam.T).T                          # [4,3]

    pts = proj[:, :2] / proj[:, 2:3]                  # [4,2] 浮点坐标

    # ★ 整体上下平移
    if y_shift_rel != 0.0:
        pts[:, 1] += y_shift_rel * h

    pts = pts.astype(int)
    origin_pt = tuple(pts[0])
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X, Y, Z (BGR)

    def in_bounds(p):
        return 0 <= p[0] < w and 0 <= p[1] < h

    for i, color in enumerate(colors):
        end_pt = tuple(pts[i + 1])
        if in_bounds(origin_pt) and in_bounds(end_pt):
            cv2.arrowedLine(img, origin_pt, end_pt, color, 2, tipLength=0.3)
    return img



def draw_3d_bbox_corners(img, R, t, K, model_points,
                         color=(0, 255, 255), thickness=1,
                         scale_x=1.0,  # 宽度方向放大系数（沿 x 轴）
                         scale_y=1.0,  # 高度方向放大系数（沿 y 轴）
                         scale_z=1.0,  # 深度方向保持不变
                         y_shift_rel=0.0):  # 如需整体上移可用负值
    """
    对 3D bbox 在物体坐标系中分别沿 x/y/z 方向缩放后再投影。
    只影响可视化，不影响真实评估和 ADD 计算。
    """
    h, w = img.shape[:2]

    pts_3d = model_points.astype(np.float32)  # [8,3]

    # 以几何中心为原点做各向异性缩放
    center = pts_3d.mean(axis=0, keepdims=True)  # [1,3]
    pts_local = pts_3d - center                  # 平移到以中心为原点

    # 构造各向异性缩放向量 [sx, sy, sz]
    scales = np.array([scale_x, scale_y, scale_z], dtype=np.float32)
    pts_scaled = pts_local * scales + center     # 放大后再平移回去

    # 应用位姿变换
    pts_cam = (R @ pts_scaled.T + t.reshape(3, 1)).T  # [8,3]
    proj = (K @ pts_cam.T).T                          # [8,3]

    # 透视投影到像素平面
    pts_2d = proj[:, :2] / proj[:, 2:3]               # [8,2]

    # 如需整体上下微调，可在像素坐标系加一个相对位移
    if y_shift_rel != 0.0:
        pts_2d[:, 1] += y_shift_rel * h

    pts_2d = pts_2d.astype(int)

    def in_bounds(p):
        return 0 <= p[0] < w and 0 <= p[1] < h

    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        p1, p2 = tuple(pts_2d[i]), tuple(pts_2d[j])
        if in_bounds(p1) and in_bounds(p2):
            cv2.line(img, p1, p2, color, thickness)
    return img




def visualize_sample(image_tensor, pred_bbox, gt_bbox, pred_R, pred_t,
                     gt_R, gt_t, K, model_points, confidence, iou,
                     rot_error, trans_error, add_dist, detected):
    """生成左(GT) 右(预测) 对比图"""
    img_base = denormalize_image(image_tensor)
    h, w = img_base.shape[:2]

    img_gt = img_base.copy()
    draw_bbox(img_gt, gt_bbox, color=(0, 255, 0), label='GT')
    img_gt = draw_3d_bbox_corners(img_gt, gt_R, gt_t, K, model_points, color=(0, 255, 0))
    img_gt = project_3d_axes(img_gt, gt_R, gt_t, K)
    cv2.putText(img_gt, 'GT', (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    img_pred = img_base.copy()
    if detected:
        bbox_color = (0, 255, 0) if iou > 0.5 else (0, 165, 255)
        draw_bbox(img_pred, pred_bbox, color=bbox_color, label=f'Pred {confidence:.2f}')
        img_pred = draw_3d_bbox_corners(img_pred, pred_R, pred_t, K, model_points, color=(0, 200, 255))
        img_pred = project_3d_axes(img_pred, pred_R, pred_t, K)
        metrics_lines = [
            f'Conf: {confidence:.3f}',
            f'IoU:  {iou:.3f}',
            f'Rot:  {rot_error:.1f}deg',
            f'Trans:{trans_error:.1f}cm',
            f'ADD:  {add_dist*100:.1f}cm',
        ]
        for i, line in enumerate(metrics_lines):
            y = h - 10 - (len(metrics_lines) - 1 - i) * 18
            cv2.putText(img_pred, line, (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        status_text = 'OK' if add_dist * 100 < 6 else 'FAIL'
        status_color = (0, 255, 0) if status_text == 'OK' else (0, 0, 255)
        cv2.putText(img_pred, status_text, (w - 60, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
    else:
        cv2.putText(img_pred, f'NOT DETECTED (conf={confidence:.3f})',
                    (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

    divider = np.full((h, 4, 3), 80, dtype=np.uint8)
    return np.concatenate([img_gt, divider, img_pred], axis=1)


def make_grid(images, cols=4, cell_size=(320, 240)):
    resized = [cv2.resize(img, cell_size) for img in images]
    rows_list = []
    for i in range(0, len(resized), cols):
        row_imgs = resized[i:i + cols]
        while len(row_imgs) < cols:
            row_imgs.append(np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8))
        rows_list.append(np.concatenate(row_imgs, axis=1))
    return np.concatenate(rows_list, axis=0)


def plot_metrics(add_list, rot_list, trans_list, iou_list, save_path):
    """逐帧折线图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Per-Sample Evaluation Metrics', fontsize=14)
    x = range(1, len(add_list) + 1)

    axes[0, 0].plot(x, [v * 100 for v in add_list], color='steelblue', lw=1)
    axes[0, 0].set_title('ADD Distance (cm)')
    axes[0, 0].set_xlabel('Sample'); axes[0, 0].set_ylabel('cm')
    axes[0, 0].axhline(y=6.0, color='red', linestyle='--', label='6cm threshold')
    axes[0, 0].legend()

    axes[0, 1].plot(x, rot_list, color='darkorange', lw=1)
    axes[0, 1].set_title('Rotation Error (°)')
    axes[0, 1].set_xlabel('Sample'); axes[0, 1].set_ylabel('degrees')
    axes[0, 1].axhline(y=5.0, color='red', linestyle='--', label='5° threshold')
    axes[0, 1].legend()

    axes[1, 0].plot(x, trans_list, color='green', lw=1)
    axes[1, 0].set_title('Translation Error (cm)')
    axes[1, 0].set_xlabel('Sample'); axes[1, 0].set_ylabel('cm')
    axes[1, 0].axhline(y=5.0, color='red', linestyle='--', label='5cm threshold')
    axes[1, 0].legend()

    axes[1, 1].plot(x, iou_list, color='purple', lw=1)
    axes[1, 1].set_title('2D BBox IoU')
    axes[1, 1].set_xlabel('Sample'); axes[1, 1].set_ylabel('IoU')
    axes[1, 1].axhline(y=0.5, color='red', linestyle='--', label='0.5 threshold')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  💾 指标折线图: {save_path}')


def plot_add_curve(add_list, threshold_m, save_path):
    """ADD AUC 曲线"""
    add_arr = np.array(add_list)
    thresholds = np.linspace(0, threshold_m * 3, 200)
    accuracies = [np.mean(add_arr < t) * 100 for t in thresholds]
    auc = np.trapz(accuracies, thresholds) / (thresholds[-1] * 100) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds * 100, accuracies, color='steelblue', lw=2)
    ax.axvline(x=threshold_m * 100, color='red', linestyle='--',
               label=f'10% diam = {threshold_m*100:.1f}cm')
    ax.fill_between(thresholds * 100, accuracies, alpha=0.15, color='steelblue')
    ax.set_xlabel('ADD Threshold (cm)'); ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'ADD AUC Curve  (AUC={auc:.1f}%)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  💾 ADD AUC 曲线: {save_path}')


def plot_trans_hist(trans_list, save_path, clip_cm=10.0):
    """
    平移误差分布直方图
    clip_cm 以上的样本忽略不计，并在标题中注明忽略数量
    """
    arr = np.array(trans_list)
    ignored = int(np.sum(arr >= clip_cm))
    arr_clip = arr[arr < clip_cm]

    fig, ax = plt.subplots(figsize=(9, 5))
    if len(arr_clip) > 0:
        bins = np.linspace(0, clip_cm, 41)  # 每 0.5cm 一个 bin
        counts, edges, patches = ax.hist(arr_clip, bins=bins,
                                         color='steelblue', edgecolor='white', linewidth=0.5)
        # 5cm 以内染绿色
        for patch, left in zip(patches, edges[:-1]):
            if left < 5.0:
                patch.set_facecolor('#2ecc71')

        ax.axvline(x=5.0,  color='red',    linestyle='--', linewidth=1.5, label='5cm')
        ax.axvline(x=3.0,  color='orange', linestyle='--', linewidth=1.5, label='3cm')
        ax.axvline(x=np.mean(arr_clip), color='navy', linestyle='-',
                   linewidth=1.5, label=f'mean={np.mean(arr_clip):.2f}cm')

        # 在图内右上角显示关键统计
        stats_text = (
            f"mean  = {np.mean(arr_clip):.2f} cm\n"
            f"median= {np.median(arr_clip):.2f} cm\n"
            f"< 3cm : {np.sum(arr_clip < 3)/len(arr_clip)*100:.1f}%\n"
            f"< 5cm : {np.sum(arr_clip < 5)/len(arr_clip)*100:.1f}%"
        )
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                fontsize=9, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=14)

    ax.set_xlabel('Translation Error (cm)')
    ax.set_ylabel('Count')
    ax.set_title(f'Translation Error Distribution')
    ax.legend(loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  💾 平移误差直方图: {save_path}')


def plot_rot_hist(rot_list, save_path, clip_deg=10.0):
    """
    旋转误差分布直方图
    clip_deg 以上的样本忽略不计，并在标题中注明忽略数量
    """
    arr = np.array(rot_list)
    ignored = int(np.sum(arr >= clip_deg))
    arr_clip = arr[arr < clip_deg]

    fig, ax = plt.subplots(figsize=(9, 5))
    if len(arr_clip) > 0:
        bins = np.linspace(0, clip_deg, 41)  # 每 0.25° 一个 bin
        counts, edges, patches = ax.hist(arr_clip, bins=bins,
                                         color='darkorange', edgecolor='white', linewidth=0.5)
        # 5° 以内染绿色
        for patch, left in zip(patches, edges[:-1]):
            if left < 5.0:
                patch.set_facecolor('#2ecc71')

        ax.axvline(x=5.0,  color='red',    linestyle='--', linewidth=1.5, label='5°')
        ax.axvline(x=3.0,  color='orange', linestyle='--', linewidth=1.5, label='3°')
        ax.axvline(x=np.mean(arr_clip), color='navy', linestyle='-',
                   linewidth=1.5, label=f'mean={np.mean(arr_clip):.2f}°')

        stats_text = (
            f"mean  = {np.mean(arr_clip):.2f}°\n"
            f"median= {np.median(arr_clip):.2f}°\n"
            f"< 3°  : {np.sum(arr_clip < 3)/len(arr_clip)*100:.1f}%\n"
            f"< 5°  : {np.sum(arr_clip < 5)/len(arr_clip)*100:.1f}%"
        )
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                fontsize=9, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=14)

    ax.set_xlabel('Rotation Error (°)')
    ax.set_ylabel('Count')
    ax.set_title(f'Rotation Error Distribution')
    ax.legend(loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  💾 旋转误差直方图: {save_path}')

# ★ 6D → 旋转矩阵（numpy 版本）
def rot6d_to_matrix_np(rot6d):
    a1 = rot6d[:3]
    a2 = rot6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)   # [3, 3]


# ─────────────────────────────────────────────────────────────
#  评估器
# ─────────────────────────────────────────────────────────────

class PoseEvaluator:
    def __init__(self, model_points, object_diameter):
        self.model_points   = model_points
        self.object_diameter= object_diameter
        self.add_threshold  = 0.1 * object_diameter  # 6cm（用于 ADD 准确率判定）
        self.iou_threshold  = 0.5                    # 用于统计 IoU>0.5 的比例
        self.conf_threshold = 0.8                    # 低于此置信度视为未检测到

    def compute_add(self, pred_R, pred_t, gt_R, gt_t):
        pred_pts = (pred_R @ self.model_points.T).T + pred_t
        gt_pts   = (gt_R   @ self.model_points.T).T + gt_t
        return np.mean(np.linalg.norm(pred_pts - gt_pts, axis=1))

    def compute_add_s(self, pred_R, pred_t, gt_R, gt_t):
        pred_pts = (pred_R @ self.model_points.T).T + pred_t
        gt_pts   = (gt_R   @ self.model_points.T).T + gt_t
        return np.mean([np.min(np.linalg.norm(gt_pts - p, axis=1)) for p in pred_pts])

    def compute_rotation_error(self, pred_R, gt_R):
        trace = np.clip(np.trace(pred_R @ gt_R.T), -1.0, 3.0)
        return np.degrees(np.arccos((trace - 1) / 2))

    def compute_translation_error(self, pred_t, gt_t):
        return np.linalg.norm(pred_t - gt_t) * 100  # cm

    def compute_iou_2d(self, bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0]); y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2]); y2 = min(bbox1[3], bbox2[3])
        if x2 < x1 or y2 < y1: return 0.0
        inter = (x2 - x1) * (y2 - y1)
        a1 = (bbox1[2]-bbox1[0]) * (bbox1[3]-bbox1[1])
        a2 = (bbox2[2]-bbox2[0]) * (bbox2[3]-bbox2[1])
        return inter / (a1 + a2 - inter + 1e-6)


# ─────────────────────────────────────────────────────────────
#  主评估函数
# ─────────────────────────────────────────────────────────────

def evaluate_model(model_path, config, use_add_s=False,
                   save_vis=True, max_vis=50, vis_every=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print(f"📊 EfficientPose 模型评估（含可视化）")
    print(f"模型: {model_path}")
    print(f"设备: {device}  |  指标: {'ADD-S' if use_add_s else 'ADD'}")
    print("=" * 70)

    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    per_sample_dir = results_dir / 'per_sample'
    if save_vis:
        per_sample_dir.mkdir(exist_ok=True)

    # ── 加载模型 ──
    test_dataset = PalletDataset(          # ← 提前创建 dataset（原来在后面）
        dataset_path=config.dataset_path,
        object_dir=config.object_dir,
        split='test', transform=None, config=config
    )
    in_ch = 4 if (test_dataset.use_lidar or test_dataset.use_depth) else 3
    model = EfficientPose(phi=config.phi, num_classes=1, in_channels=in_ch).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    print(f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'state_dict'}")

    if isinstance(checkpoint, dict):
        key = next((k for k in ['model_state_dict', 'model', 'state_dict'] if k in checkpoint), None)
        model.load_state_dict(checkpoint[key] if key else checkpoint)
        epoch = checkpoint.get('epoch', 'N/A') if key else 'N/A'
    else:
        model.load_state_dict(checkpoint)
        epoch = 'N/A'

    model.eval()
    print(f"✅ 加载模型成功  Epoch: {epoch}")

    # ── 数据集 ──

    print(f"✅ 测试集大小: {len(test_dataset)} 张")

    model_points    = config.get_model_points().numpy()
    object_diameter = 0.6
    evaluator       = PoseEvaluator(model_points, object_diameter)

    metrics = {
        'detection_success': 0,
        'add_distances':     [],
        'rotation_errors':   [],
        'translation_errors':[],
        'iou_2d':            [],
        'confidences':       [],
    }
    grid_frames = []
    vis_count   = 0

    with torch.no_grad():
        # 诊断第一帧
        if len(test_dataset) > 0:
            print("\n🔍 诊断第一个样本的 anchor 分布...")
            s = test_dataset[0]
            scores = model(s['image'].unsqueeze(0).to(device))['class'][0, :, 0].cpu().numpy()
            print(f"  置信度 > 0.5: {np.sum(scores > 0.5)} / {len(scores)}")
            print(f"  最高置信度: {scores.max():.4f}  Top-5: {np.sort(scores)[-5:][::-1]}\n")

        for idx in tqdm(range(len(test_dataset)), desc="评估进度"):
            sample        = test_dataset[idx]
            image_tensor  = sample['image']
            gt_bbox       = sample['bbox'].numpy()
            gt_rotation   = sample['rotation'].numpy()
            gt_translation= sample['translation'].numpy()
            K = sample['camera_matrix'].numpy().copy()
            orig_h, orig_w = 480, 640          # 相机原始分辨率
            target_size = config.image_size    # 512

            scale_x = target_size / orig_w    # 512/640 = 0.8
            scale_y = target_size / orig_h    # 512/480 ≈ 1.0667

            K[0, 0] *= scale_x   # fx
            K[1, 1] *= scale_y   # fy
            K[0, 2] *= scale_x   # cx
            K[1, 2] *= scale_y   # cy

            preds      = model(image_tensor.unsqueeze(0).to(device))
            scores     = preds['class'][0, :, 0].cpu().numpy()
            best_idx   = np.argmax(scores)
            confidence = float(scores[best_idx])
            metrics['confidences'].append(confidence)

            detected      = confidence >= evaluator.conf_threshold
            pred_bbox     = np.array([0., 0., 0., 0.])
            pred_rotation = np.eye(3)
            pred_t        = np.zeros(3)
            iou = rot_error = trans_error = add_dist = 0.0

            if detected:
                metrics['detection_success'] += 1
                pred_bbox = np.clip(preds['bbox'][0, best_idx].cpu().numpy(), 0.0, 1.0)
                pred_t    = preds['translation'][0, best_idx].cpu().numpy()
                try:
                    rot6d = preds['rotation'][0, best_idx].cpu().numpy()   # [6]
                    pred_rotation = rot6d_to_matrix_np(rot6d)
                except np.linalg.LinAlgError:
                    metrics['detection_success'] -= 1
                    detected = False
                    pred_rotation = np.eye(3)

            if detected:
                add_dist    = (evaluator.compute_add_s if use_add_s else evaluator.compute_add)(
                                  pred_rotation, pred_t, gt_rotation, gt_translation)
                rot_error   = evaluator.compute_rotation_error(pred_rotation, gt_rotation)
                trans_error = evaluator.compute_translation_error(pred_t, gt_translation)
                iou         = evaluator.compute_iou_2d(pred_bbox, gt_bbox)

                metrics['add_distances'].append(add_dist)
                metrics['rotation_errors'].append(rot_error)
                metrics['translation_errors'].append(trans_error)
                metrics['iou_2d'].append(iou)

            if save_vis and (idx % vis_every == 0) and vis_count < max_vis:
                vis_img = visualize_sample(
                    image_tensor, pred_bbox, gt_bbox,
                    pred_rotation, pred_t, gt_rotation, gt_translation,
                    K, model_points, confidence, iou,
                    rot_error, trans_error, add_dist, detected
                )
                cv2.imwrite(str(per_sample_dir / f'sample_{idx:04d}.jpg'), vis_img)
                half_w = vis_img.shape[1] // 2
                grid_frames.append(vis_img[:, half_w + 4:])
                vis_count += 1

    # ── 保存图像输出 ──
    print("\n📊 生成可视化图表...")

    if grid_frames:
        grid_img = make_grid(grid_frames, cols=4, cell_size=(320, 240))
        grid_path = results_dir / 'eval_grid.jpg'
        cv2.imwrite(str(grid_path), grid_img)
        print(f"  🖼️  汇总网格图 ({len(grid_frames)} 张): {grid_path}")

    if metrics['add_distances']:
        plot_metrics(
            metrics['add_distances'], metrics['rotation_errors'],
            metrics['translation_errors'], metrics['iou_2d'],
            save_path=str(results_dir / 'eval_metrics.png')
        )
        plot_add_curve(
            metrics['add_distances'], evaluator.add_threshold,
            save_path=str(results_dir / 'eval_add_curve.png')
        )
        plot_trans_hist(
            metrics['translation_errors'],
            save_path=str(results_dir / 'eval_trans_hist.png'),
            clip_cm=20.0   # ← 20cm 以上忽略
        )
        plot_rot_hist(
            metrics['rotation_errors'],
            save_path=str(results_dir / 'eval_rot_hist.png'),
            clip_deg=10.0  # ← 10° 以上忽略
        )

    # ── 汇总统计 ──
    print("\n" + "=" * 70)
    print("📈 评估结果")
    print("=" * 70)

    total      = len(test_dataset)
    detected_n = metrics['detection_success']
    print(f"\n【检测性能】")
    print(f"  总样本数:   {total}")
    print(f"  检测成功:   {detected_n} ({detected_n/total*100:.2f}%)")
    print(f"  平均置信度: {np.mean(metrics['confidences']):.4f}")

    if detected_n > 0:
        add_arr   = np.array(metrics['add_distances'])
        rot_arr   = np.array(metrics['rotation_errors'])
        trans_arr = np.array(metrics['translation_errors'])
        iou_arr   = np.array(metrics['iou_2d'])
        add_acc   = np.sum(add_arr < evaluator.add_threshold) / len(add_arr) * 100

        print(f"\n【{'ADD-S' if use_add_s else 'ADD'} 指标】(阈值: {evaluator.add_threshold*100:.1f} cm)")
        print(f"  平均 ADD:   {np.mean(add_arr)*100:.2f} cm")
        print(f"  中位数 ADD: {np.median(add_arr)*100:.2f} cm")
        print(f"  ADD 准确率: {add_acc:.2f}%")

        print(f"\n【旋转误差】(全部样本)")
        print(f"  平均:  {np.mean(rot_arr):.2f}°  |  中位数: {np.median(rot_arr):.2f}°")
        print(f"  < 5°:  {np.sum(rot_arr < 5) / len(rot_arr) * 100:.2f}%")
        print(f"  < 10°: {np.sum(rot_arr < 10) / len(rot_arr) * 100:.2f}%")
        ignored_rot = int(np.sum(rot_arr >= 10))
        print(f"  ≥ 10° (直方图忽略): {ignored_rot} 张")

        print(f"\n【平移误差】(全部样本)")
        print(f"  平均:  {np.mean(trans_arr):.2f} cm  |  中位数: {np.median(trans_arr):.2f} cm")
        print(f"  < 2cm: {np.sum(trans_arr < 2) / len(trans_arr) * 100:.2f}%")
        print(f"  < 5cm: {np.sum(trans_arr < 5) / len(trans_arr) * 100:.2f}%")
        ignored_trans = int(np.sum(trans_arr >= 20))
        print(f"  ≥ 20cm (直方图忽略): {ignored_trans} 张")

        print(f"\n【2D 检测】")
        print(f"  平均 IoU: {np.mean(iou_arr):.4f}")
        print(f"  IoU > 0.5: {np.sum(iou_arr > 0.5) / len(iou_arr) * 100:.2f}%")
        print(f"  IoU > 0.75: {np.sum(iou_arr > 0.75) / len(iou_arr) * 100:.2f}%")

        print(f"\n【综合评价】")
        if add_acc > 90 and np.mean(trans_arr) < 5:
            print("  ✅ 优秀 - 可用于生产部署")
        elif add_acc > 75 and np.mean(trans_arr) < 10:
            print("  🟡 良好 - 建议继续训练")
        else:
            print("  ❌ 需要改进 - 检查数据或模型")

        results_path = results_dir / 'eval_results.txt'
        with open(results_path, 'w') as f:
            f.write(f"模型: {model_path}\n")
            f.write(f"检测率: {detected_n/total*100:.2f}%\n")
            f.write(f"平均 ADD: {np.mean(add_arr)*100:.2f} cm\n")
            f.write(f"ADD 准确率: {add_acc:.2f}%\n")
            f.write(f"平均旋转误差: {np.mean(rot_arr):.2f}°\n")
            f.write(f"平均平移误差: {np.mean(trans_arr):.2f} cm\n")
        print(f"\n💾 文本结果: {results_path}")

    print(f"\n📁 所有图像输出目录: {results_dir}")
    print("=" * 70)


if __name__ == '__main__':
    config = Config()
    checkpoints_dir = Path(config.checkpoints_dir)
    model_files = sorted(checkpoints_dir.glob('best_model_phi*.pth'))

    if not model_files:
        print(f"❌ 错误: {checkpoints_dir} 中没有找到模型文件")
        sys.exit(1)

    evaluate_model(
        model_path=str(model_files[0]),
        config=config,
        use_add_s=False,
        save_vis=True,
        max_vis=50,
        vis_every=1
    )

