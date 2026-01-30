"""
Panorama Generation - Cylindrical Projection + Z-buffer
"""

import numpy as np
from numba import jit
from typing import List, Tuple, Optional
from config import Config, PANO_WIDTH, PANO_HEIGHT


class PanoramaRenderer:
    """
    Panorama Renderer

    Uses cylindrical projection to render point clouds into panoramic images
    Supports RGB, depth, and segmentation channels
    """

    def __init__(
        self,
        width: int = PANO_WIDTH,
        height: int = PANO_HEIGHT,
        view_dist: float = 10.0
    ):
        """
        Args:
            width: Panorama width (default 360)
            height: Panorama height (default 180)
            view_dist: Maximum viewing distance (meters)
        """
        self.width = width
        self.height = height
        self.view_dist = view_dist

        # Cylindrical projection parameters
        self.u_step = 1.0  # Horizontal resolution (degrees/pixel)
        self.v_step = 1.0  # Vertical resolution (degrees/pixel)
        self.u_min = 0
        self.u_max = width
        self.v_min = -height // 2
        self.v_max = height // 2

    def render_depth_only(
        self,
        pc_frames: List[np.ndarray],
        curr_pos: np.ndarray,
        yaw_matrix: np.ndarray,
        filter_dist: bool = True
    ) -> np.ndarray:
        """
        Render depth panorama (single channel)

        Args:
            pc_frames: List of point cloud frames, each frame is (N, 3+) array
            curr_pos: Current position (3,)
            yaw_matrix: Yaw rotation matrix (3, 3)
            filter_dist: Whether to filter by distance

        Returns:
            (H, W) depth panorama, range 0-255
        """
        # Merge all point clouds
        if len(pc_frames) == 0:
            return np.ones((self.height, self.width), dtype=np.uint8) * 255

        total_pts = np.vstack([pc[:, :3] for pc in pc_frames if len(pc) > 0])
        if len(total_pts) == 0:
            return np.ones((self.height, self.width), dtype=np.uint8) * 255

        # Transform to local coordinate system
        pts_local = total_pts - curr_pos
        pts_local = pts_local @ yaw_matrix

        x, y, z = pts_local[:, 0], pts_local[:, 1], pts_local[:, 2]

        # Calculate distance
        dxy = np.sqrt(x**2 + y**2)  # Horizontal distance
        dist = np.sqrt(dxy**2 + z**2)  # 3D distance

        # Distance filtering
        if filter_dist:
            mask = dist <= self.view_dist
            x, y, z = x[mask], y[mask], z[mask]
            dxy, dist = dxy[mask], dist[mask]

        if len(x) == 0:
            return np.ones((self.height, self.width), dtype=np.uint8) * 255

        # Cylindrical projection
        u, v = self._lidar_to_surround_coords(x, y, z, dxy)

        # Normalize depth to 0-255
        depth_norm = self._normalize_depth(dist)

        # Render (z-buffer)
        pano = np.ones((self.height, self.width), dtype=np.uint8) * 255
        self._render_zbuffer(pano, u, v, depth_norm)

        return pano

    def render_with_color(
        self,
        pc_frames: List[np.ndarray],
        curr_pos: np.ndarray,
        yaw_matrix: np.ndarray,
        filter_dist: bool = True,
        sample_ratio: float = 1.0
    ) -> np.ndarray:
        """
        Render RGB+D panorama

        Args:
            pc_frames: List of point cloud frames, each frame is (N, 6+) array (xyz + rgb + ...)
            curr_pos: Current position
            yaw_matrix: Yaw rotation matrix
            filter_dist: Whether to filter by distance
            sample_ratio: Point cloud downsampling ratio

        Returns:
            (H, W, 4) RGBD panorama
        """
        # Merge all point clouds
        if len(pc_frames) == 0:
            pano = np.zeros((self.height, self.width, 4), dtype=np.float32)
            pano[:, :, 3] = 255  # Initialize depth to maximum
            return pano

        # Extract valid frames
        valid_frames = [pc for pc in pc_frames if len(pc) > 0 and pc.shape[1] >= 6]
        if len(valid_frames) == 0:
            pano = np.zeros((self.height, self.width, 4), dtype=np.float32)
            pano[:, :, 3] = 255
            return pano

        total_pts = np.vstack(valid_frames)

        # Downsampling
        if sample_ratio < 1.0:
            n_keep = int(len(total_pts) * sample_ratio)
            if n_keep < len(total_pts):
                indices = np.random.choice(len(total_pts), n_keep, replace=False)
                total_pts = total_pts[indices]

        # Transform to local coordinate system
        pts_local = total_pts.copy()
        pts_local[:, :3] = pts_local[:, :3] - curr_pos
        pts_local[:, :3] = pts_local[:, :3] @ yaw_matrix

        x, y, z = pts_local[:, 0], pts_local[:, 1], pts_local[:, 2]
        r, g, b = pts_local[:, 3], pts_local[:, 4], pts_local[:, 5]

        # Calculate distance
        dxy = np.sqrt(x**2 + y**2)
        dist = np.sqrt(dxy**2 + z**2)

        # Distance filtering
        if filter_dist:
            mask = dist <= self.view_dist
            x, y, z = x[mask], y[mask], z[mask]
            r, g, b = r[mask], g[mask], b[mask]
            dxy, dist = dxy[mask], dist[mask]

        if len(x) == 0:
            pano = np.zeros((self.height, self.width, 4), dtype=np.float32)
            pano[:, :, 3] = 255
            return pano

        # Cylindrical projection
        u, v = self._lidar_to_surround_coords(x, y, z, dxy)

        # Normalize depth
        depth_norm = self._normalize_depth(dist)

        # Render (z-buffer)
        pano = np.zeros((self.height, self.width, 4), dtype=np.float32)
        pano[:, :, 3] = 255  # Initialize depth to maximum

        self._render_rgbd_zbuffer(pano, u, v, r, g, b, depth_norm)

        return pano

    def render_with_seg(
        self,
        pc_frames: List[np.ndarray],
        curr_pos: np.ndarray,
        yaw_matrix: np.ndarray,
        filter_dist: bool = True,
        sample_ratio: float = 0.4
    ) -> np.ndarray:
        """
        Render RGB+D+Seg panorama

        Args:
            pc_frames: List of point cloud frames, each frame is (N, 13) array (xyz + rgb + 7ch_seg)
            curr_pos: Current position
            yaw_matrix: Yaw rotation matrix

        Returns:
            (H, W, 11) panorama (RGB + D + 7ch_seg)
        """
        if len(pc_frames) == 0:
            pano = np.zeros((self.height, self.width, 11), dtype=np.float32)
            pano[:, :, 3] = 255
            return pano

        # Merge point clouds
        valid_frames = [pc[0] if isinstance(pc, list) else pc
                       for pc in pc_frames if len(pc) > 0]
        if len(valid_frames) == 0:
            pano = np.zeros((self.height, self.width, 11), dtype=np.float32)
            pano[:, :, 3] = 255
            return pano

        total_pts = np.concatenate(valid_frames, axis=0)

        # Downsampling
        if sample_ratio < 1.0:
            n_keep = int(len(total_pts) * sample_ratio)
            if n_keep < len(total_pts):
                indices = np.random.choice(len(total_pts), n_keep, replace=False)
                total_pts = total_pts[indices]

        # Transform to local coordinate system
        pts_local = total_pts.copy()
        pts_local[:, :3] = pts_local[:, :3] - curr_pos
        pts_local[:, :3] = pts_local[:, :3] @ yaw_matrix

        x, y, z = pts_local[:, 0], pts_local[:, 1], pts_local[:, 2]

        # Calculate distance
        dxy = np.sqrt(x**2 + y**2)
        dist = np.sqrt(dxy**2 + z**2)

        # Distance filtering
        if filter_dist:
            mask = dist <= self.view_dist
            pts_local = pts_local[mask]
            x, y, z = x[mask], y[mask], z[mask]
            dxy, dist = dxy[mask], dist[mask]

        if len(x) == 0:
            pano = np.zeros((self.height, self.width, 11), dtype=np.float32)
            pano[:, :, 3] = 255
            return pano

        # Cylindrical projection
        u, v = self._lidar_to_surround_coords(x, y, z, dxy)

        # Normalize depth
        depth_norm = self._normalize_depth(dist)

        # Render
        pano = np.zeros((self.height, self.width, 11), dtype=np.float32)
        pano[:, :, 3] = 255

        self._render_full_zbuffer(pano, u, v, depth_norm, pts_local)

        return pano

    def _lidar_to_surround_coords(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        dxy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Point cloud coordinates -> Panorama pixel coordinates

        Cylindrical projection:
        u = arctan2(x, y) -> Horizontal angle
        v = arctan2(z, dxy) -> Vertical angle
        """
        # Horizontal angle (degrees)
        u = np.arctan2(x, y) / np.pi * 180 / self.u_step
        # Vertical angle (degrees)
        v = -np.arctan2(z, dxy) / np.pi * 180 / self.v_step

        # Adjust u to [0, 360)
        u = (u + 90 + 360) % 360

        # Convert to pixel coordinates
        u = np.floor(u).astype(np.uint16)
        v = np.floor(v - self.v_min).astype(np.uint16)

        # Boundary clipping
        u = np.clip(u, 0, self.width - 1)
        v = np.clip(v, 0, self.height - 1)

        return u, v

    def _normalize_depth(self, dist: np.ndarray) -> np.ndarray:
        """Normalize depth to 0-255"""
        # dist / 10.0 * 255.0, i.e., 10 meters corresponds to 255
        depth_clipped = np.clip(dist, 0, self.view_dist)
        return (depth_clipped / self.view_dist * 255.0).astype(np.uint8)

    @staticmethod
    @jit(nopython=True)
    def _render_zbuffer(pano: np.ndarray, u: np.ndarray, v: np.ndarray, d: np.ndarray):
        """Z-buffer rendering (depth priority)"""
        for i in range(len(u)):
            if pano[v[i], u[i]] > d[i]:
                pano[v[i], u[i]] = d[i]

    @staticmethod
    @jit(nopython=True)
    def _render_rgbd_zbuffer(
        pano: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        r: np.ndarray,
        g: np.ndarray,
        b: np.ndarray,
        d: np.ndarray
    ):
        """Z-buffer rendering for RGBD"""
        for i in range(len(u)):
            if pano[v[i], u[i], 3] > d[i]:
                pano[v[i], u[i], 0] = r[i]
                pano[v[i], u[i], 1] = g[i]
                pano[v[i], u[i], 2] = b[i]
                pano[v[i], u[i], 3] = d[i]

    @staticmethod
    @jit(nopython=True)
    def _render_full_zbuffer(
        pano: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        d: np.ndarray,
        pts: np.ndarray
    ):
        """Z-buffer rendering for full data (RGB + D + 7ch_seg)"""
        for i in range(len(u)):
            if pano[v[i], u[i], 3] > d[i]:
                # RGB
                pano[v[i], u[i], 0] = pts[i, 3]
                pano[v[i], u[i], 1] = pts[i, 4]
                pano[v[i], u[i], 2] = pts[i, 5]
                # Depth
                pano[v[i], u[i], 3] = d[i]
                # Segmentation (7 channels)
                for j in range(7):
                    pano[v[i], u[i], 4 + j] = pts[i, 6 + j]

    def to_5_channel(self, pano_11ch: np.ndarray) -> np.ndarray:
        """
        Convert 11-channel panorama to 5-channel (R, G, B, D, Seg_intensity)

        Args:
            pano_11ch: (H, W, 11) panorama

        Returns:
            (H, W, 5) panorama
        """
        pano_5ch = np.zeros((self.height, self.width, 5), dtype=np.float32)

        # RGB + D
        pano_5ch[:, :, :4] = pano_11ch[:, :, :4]

        # Seg: argmax of 7 channels -> intensity (0-7)
        seg_7ch = pano_11ch[:, :, 4:11]
        seg_sum = np.sum(seg_7ch, axis=-1)
        seg_int = np.argmax(seg_7ch, axis=-1)
        seg_int[seg_sum == 0] = 7  # No label -> 7 (other)
        pano_5ch[:, :, 4] = seg_int

        return pano_5ch
