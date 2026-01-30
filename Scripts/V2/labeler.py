"""
Semantic Segmentation - DINOv2 + Mask2Former
"""

import numpy as np
from typing import Optional, List, Dict
from tqdm import tqdm

from config import ADE2OUR, NUM_CLASSES


class SemanticLabeler:
    """
    Semantic Segmenter

    Uses DINOv2 + Mask2Former for semantic segmentation
    Maps ADE20K 150 classes to 8 classes
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.model = None
        self._loaded = False

    def load_model(self):
        """Load DINOv2 segmentation model"""
        if self._loaded:
            return

        try:
            import sys
            import torch
            import mmcv
            import urllib
            from mmcv.runner import load_checkpoint
            from mmseg.apis import init_segmentor

            # Add dinov2 repo to path and register model
            DINOV2_REPO = "/afs/cs.stanford.edu/u/weizhuo2/Documents/gits/dinov2"
            if DINOV2_REPO not in sys.path:
                sys.path.append(DINOV2_REPO)
            import dinov2.eval.segmentation_m2f.models.segmentors

            # DINOv2 configuration and weights
            DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
            CONFIG_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f_config.py"
            CHECKPOINT_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f.pth"

            print("Loading DINOv2 config...")
            with urllib.request.urlopen(CONFIG_URL) as f:
                cfg_str = f.read().decode()

            cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

            # Optimization: use 'whole' mode for faster inference (600ms -> 380ms)
            cfg.model.test_cfg.mode = "whole"

            print("Loading DINOv2 model...")
            self.model = init_segmentor(cfg)
            load_checkpoint(self.model, CHECKPOINT_URL, map_location="cpu")
            self.model.to(self.device)
            self.model.eval()

            self._loaded = True
            print("DINOv2 model loaded successfully")

        except ImportError as e:
            print(f"Warning: Cannot load DINOv2 model: {e}")
            print("Please install: pip install mmcv mmseg dinov2")
            raise

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Perform semantic segmentation on a single image

        Args:
            image_bgr: (H, W, 3) BGR image

        Returns:
            (H, W, 7) one-hot segmentation result
        """
        if not self._loaded:
            self.load_model()

        from mmseg.apis import inference_segmentor

        # Inference
        seg_ade20k = inference_segmentor(self.model, image_bgr)[0]

        # ADE20K -> 8 classes
        seg_cls = ADE2OUR[seg_ade20k + 1].astype(np.uint8)

        # Convert to one-hot (7 channels, excluding "other")
        H, W = seg_cls.shape
        seg_onehot = np.zeros((H, W, 7), dtype=np.float32)
        for i in range(7):
            seg_onehot[:, :, i] = (seg_cls == i).astype(np.float32)

        return seg_onehot

    def segment_batch(self, images_bgr: List[np.ndarray]) -> List[np.ndarray]:
        """
        Batch segmentation

        Args:
            images_bgr: List of BGR images

        Returns:
            List of segmentation results
        """
        results = []
        for img in tqdm(images_bgr, desc="Segmenting"):
            seg = self.segment(img)
            results.append(seg)
        return results

    def ade20k_to_classes(self, seg_ade: np.ndarray) -> np.ndarray:
        """
        Convert ADE20K labels to 8 classes

        Args:
            seg_ade: (H, W) ADE20K segmentation result (0-149)

        Returns:
            (H, W) 8-class segmentation result (0-7)
        """
        return ADE2OUR[seg_ade + 1]

    def to_intensity(self, seg_onehot: np.ndarray) -> np.ndarray:
        """
        Convert one-hot to intensity labels

        Args:
            seg_onehot: (H, W, 7) one-hot segmentation

        Returns:
            (H, W) intensity labels (0-7)
        """
        seg_sum = np.sum(seg_onehot, axis=-1)
        seg_int = np.argmax(seg_onehot, axis=-1)
        seg_int[seg_sum == 0] = 7  # No label -> other
        return seg_int.astype(np.uint8)


class SimpleLabelProjector:
    """
    Simple Label Projector

    Projects 2D segmentation results to point cloud
    """

    def __init__(self, intrinsics):
        self.intrinsics = intrinsics

    def project_seg_to_pointcloud(
        self,
        depth: np.ndarray,
        seg: np.ndarray,
        color: np.ndarray = None,
        min_depth: float = 0.3
    ) -> np.ndarray:
        """
        Project segmentation results to point cloud

        Args:
            depth: (H, W) depth map (meters)
            seg: (H, W, 7) segmentation result
            color: (H, W, 3) RGB image (optional)
            min_depth: minimum depth

        Returns:
            (N, 13) point cloud [x, y, z, r, g, b, c1, c2, c3, c4, c5, c6, c7]
        """
        H, W = depth.shape
        fx, fy = self.intrinsics.fx, self.intrinsics.fy
        cx, cy = self.intrinsics.cx, self.intrinsics.cy

        # Pixel grid
        v, u = np.indices((H, W))

        # Valid depth
        mask = depth > min_depth

        u_valid = u[mask]
        v_valid = v[mask]
        z = depth[mask]

        # Back-projection
        x = (u_valid - cx) * z / fx
        y = (v_valid - cy) * z / fy

        # Color
        if color is not None:
            rgb = color[v_valid, u_valid, :3]
        else:
            rgb = np.zeros((len(z), 3))

        # Segmentation
        seg_vals = seg[v_valid, u_valid, :]

        # Combine
        points = np.column_stack([x, y, z, rgb, seg_vals])
        return points.astype(np.float32)


# Semantic class visualization colors (matches original prompts order)
CLASS_COLORS = [
    [0.1, 0.1, 0.1, 0.7],    # 0: floor - dark
    [0.7, 0.7, 0.7, 0.7],    # 1: stair - gray
    [0.0, 0.7, 0.0, 0.7],    # 2: door - green
    [0.7, 0.0, 0.0, 0.7],    # 3: wall - red
    [0.0, 0.0, 0.7, 0.7],    # 4: obstacle - blue
    [0.05, 0.7, 0.7, 0.7],   # 5: human - cyan
    [0.5, 0.5, 0.5, 0.7],    # 6: terrain - gray
    [1.0, 1.0, 1.0, 0.3],    # 7: other - white (transparent)
]


def visualize_segmentation(image: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """
    Visualize segmentation results

    Args:
        image: (H, W, 3) RGB or BGR image
        seg: (H, W, 7) one-hot segmentation or (H, W) intensity segmentation

    Returns:
        (H, W, 4) RGBA overlay image
    """
    from PIL import Image as PILImage

    H, W = image.shape[:2]

    # Convert to intensity if one-hot
    if seg.ndim == 3:
        seg_int = np.argmax(seg, axis=-1)
        seg_int[np.sum(seg, axis=-1) == 0] = 7
    else:
        seg_int = seg

    # Create overlay image
    overlay = np.zeros((H, W, 4), dtype=np.float32)

    for cls_id in range(len(CLASS_COLORS)):
        mask = seg_int == cls_id
        color = CLASS_COLORS[cls_id]
        overlay[mask] = color

    # Blend
    base = image.astype(np.float32) / 255.0
    if base.shape[2] == 3:
        base = np.concatenate([base, np.ones((H, W, 1))], axis=-1)

    alpha = overlay[:, :, 3:4]
    result = base * (1 - alpha) + overlay[:, :, :3] * alpha
    result = np.concatenate([result[:, :, :3], np.ones((H, W, 1))], axis=-1)

    return (result * 255).astype(np.uint8)
