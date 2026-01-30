"""
Depth Processing - Point Cloud to Depth Map, Edge Removal
"""

import numpy as np
import cv2
from typing import Optional
from config import Config, D455_INTRINSICS


class DepthProcessor:
    """
    Depth Map Processor

    Features:
    - Point Cloud -> Depth Map (reprojection)
    - Edge Removal (Canny + erode)
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.intrinsics = self.config.intrinsics

    def pointcloud_to_depth(
        self,
        pc: np.ndarray,
        min_depth: float = 0.3
    ) -> np.ndarray:
        """
        Reproject point cloud to depth map

        Corresponds to original code point_to_channelD455()

        Args:
            pc: (N, 3+) point cloud, xyz in camera coordinate system
            min_depth: minimum valid depth

        Returns:
            (H, W, 1) depth map, normalized to [0, 1] (10m = 1.0)
        """
        fx, fy = self.intrinsics.fx, self.intrinsics.fy
        cx, cy = self.intrinsics.cx, self.intrinsics.cy
        W, H = self.intrinsics.width, self.intrinsics.height

        # Initialize depth map
        depth_frame = np.zeros((H, W, 1), dtype=np.float32)

        # Filter invalid points
        mask = pc[:, 2] > min_depth
        pc = pc[mask]

        if len(pc) == 0:
            return depth_frame

        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]

        # Project to pixel coordinates
        u = np.round(fx * x / z + cx).astype(int)
        v = np.round(fy * y / z + cy).astype(int)

        # Boundary check
        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, z = u[valid], v[valid], z[valid]

        # Normalize depth (10m = 1.0)
        depth_norm = z / 10.0

        # Fill depth map (simple overwrite, no z-buffer)
        depth_frame[v, u, 0] = depth_norm

        return depth_frame

    def remove_edges(
        self,
        depth: np.ndarray,
        canny_low: int = 25,
        canny_high: int = 32,
        kernel_size: int = 10
    ) -> np.ndarray:
        """
        Remove depth map edges (Canny + erode)

        Corresponds to original code remove_edges()

        Depth values at edges are usually inaccurate, this function sets edge region depths to 0

        Args:
            depth: (H, W) or (H, W, 1) depth map
            canny_low: Canny edge detection low threshold
            canny_high: Canny edge detection high threshold
            kernel_size: edge expansion kernel size

        Returns:
            processed depth map
        """
        # Ensure input format
        if depth.ndim == 2:
            depth = depth[:, :, np.newaxis]

        depth_out = depth.copy()
        H, W, C = depth_out.shape

        # Convert to uint8 for edge detection
        depth_uint8 = (np.clip(depth_out[:, :, 0], 0, 1) * 255).astype(np.uint8)

        # Canny edge detection
        edges = cv2.Canny(depth_uint8, canny_low, canny_high)

        # Expand edges (erode)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        edges_expanded = cv2.dilate(edges, kernel)

        # Create mask (edge regions)
        mask = edges_expanded > 0

        # Set edge region depths to 0
        for c in range(C):
            depth_out[:, :, c][mask] = 0

        return depth_out

    def fill_holes(
        self,
        depth: np.ndarray,
        kernel_size: int = 5
    ) -> np.ndarray:
        """
        Fill depth map holes (optional)

        Use morphological closing operation to fill small holes

        Args:
            depth: (H, W) or (H, W, 1) depth map
            kernel_size: morphological operation kernel size

        Returns:
            filled depth map
        """
        if depth.ndim == 3:
            depth_2d = depth[:, :, 0]
        else:
            depth_2d = depth

        depth_out = depth_2d.copy()

        # Convert to uint8
        depth_uint8 = (np.clip(depth_out, 0, 1) * 255).astype(np.uint8)

        # Morphological closing operation
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        depth_closed = cv2.morphologyEx(depth_uint8, cv2.MORPH_CLOSE, kernel)

        # Convert back to float
        depth_out = depth_closed.astype(np.float32) / 255.0

        if depth.ndim == 3:
            return depth_out[:, :, np.newaxis]
        return depth_out


def fixrgb(rnp_pc: np.ndarray) -> np.ndarray:
    """
    Fix point cloud RGB values converted by ros_numpy

    ros_numpy packs RGB as float32, needs to be unpacked to uint8

    Args:
        rnp_pc: structured point cloud array converted by ros_numpy

    Returns:
        (N, 6) array [x, y, z, r, g, b]
    """
    # Flatten
    if len(rnp_pc.shape) == 2:
        rnp_pc = rnp_pc.flatten()

    # Extract xyz
    xyz = np.column_stack((rnp_pc['x'], rnp_pc['y'], rnp_pc['z']))

    # Unpack rgb
    float_lst = rnp_pc['rgb'].copy()
    byte_repr = float_lst.view(np.uint8).reshape(-1, 4)
    rgb = byte_repr[:, :3].astype(float)  # BGR order

    return np.column_stack((xyz, rgb))
