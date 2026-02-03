"""
Panorama Generation - Cylindrical Projection + Z-buffer
"""

import numpy as np
from typing import List, Tuple
from config import PANO_WIDTH, PANO_HEIGHT


class PanoramaRenderer:
    """
    Panorama Renderer

    Uses cylindrical projection to render point clouds into panoramic images.
    Supports arbitrary number of channels.
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
        self.v_min = -height // 2
        self.v_max = height // 2

    def render(
        self,
        pc_frames: List[np.ndarray],
        curr_pos: np.ndarray,
        yaw_matrix: np.ndarray,
        filter_dist: bool = True,
        sample_ratio: float = 1.0
    ) -> np.ndarray:
        """
        Render panorama from point clouds.

        Args:
            pc_frames: List of point cloud frames, each is (N, C) array
                       - [:, 0:3] = xyz coordinates
                       - [:, 3:] = additional data (rgb, seg, etc.)
            curr_pos: Current position (3,)
            yaw_matrix: Yaw rotation matrix (3, 3)
            filter_dist: Whether to filter by distance
            sample_ratio: Point cloud downsampling ratio

        Returns:
            (H, W, C-3+1) panorama
            - If C > 3: [data[:, 3:], depth] → matches original pipeline [RGB, D, seg]
            - If C == 3: [depth] only
        """
        # Merge point clouds
        valid_frames = []
        for pc in pc_frames:
            if isinstance(pc, list):
                pc = pc[0] if len(pc) > 0 else np.array([])
            if len(pc) > 0:
                valid_frames.append(pc)

        if len(valid_frames) == 0:
            # Return empty panorama (depth only)
            pano = np.ones((self.height, self.width, 1), dtype=np.float32) * 255
            return pano

        total_pts = np.concatenate(valid_frames, axis=0)
        n_channels = total_pts.shape[1]
        extra_channels = n_channels - 3  # Channels beyond xyz

        # Downsampling
        if sample_ratio < 1.0 and len(total_pts) > 0:
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
            pano = np.ones((self.height, self.width, 1 + extra_channels), dtype=np.float32) * 255
            if extra_channels > 0:
                pano[:, :, 1:] = 0
            return pano

        # Cylindrical projection
        u, v = self._project(x, y, z, dxy)

        # Normalize depth
        depth_norm = self._normalize_depth(dist)

        # Render with z-buffer
        # Output format: [extra_data, depth] to match original pipeline [RGB, D, seg]
        # If no extra data, output is [depth] only
        if extra_channels > 0:
            out_channels = extra_channels + 1  # [RGB(3), D(1), seg(7)] = 11
            depth_idx = 3 if extra_channels >= 3 else extra_channels  # D after RGB
        else:
            out_channels = 1
            depth_idx = 0

        pano = np.zeros((self.height, self.width, out_channels), dtype=np.float32)
        pano[:, :, depth_idx] = 255  # Depth initialized to max (unknown)

        self._zbuffer_render(pano, u, v, depth_norm, pts_local, depth_idx)

        return pano

    def _project(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        dxy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cylindrical projection: xyz -> (u, v) pixel coordinates

        Same as original pipeline lidar_to_surround_coords()
        """
        # Horizontal angle (degrees)
        u = np.arctan2(x, y) / np.pi * 180 / self.u_step
        # Vertical angle (degrees)
        v = -np.arctan2(z, dxy) / np.pi * 180 / self.v_step

        # Adjust u to [0, width)
        u = (u + 90 + 360) % 360

        # Convert to pixel coordinates (same as original: floor then cast)
        u = np.floor(u).astype(np.uint16)
        v = np.floor(v - self.v_min).astype(np.uint16)

        # Clip to valid range (original doesn't clip, but uint16 overflow can cause issues)
        u = np.clip(u, 0, self.width - 1)
        v = np.clip(v, 0, self.height - 1)

        return u, v

    def _normalize_depth(self, dist: np.ndarray) -> np.ndarray:
        """Normalize depth to 0-255 (0=close, 255=far)"""
        depth_clipped = np.clip(dist, 0, self.view_dist)
        return (depth_clipped / self.view_dist * 255.0).astype(np.float32)

    def _zbuffer_render(
        self,
        pano: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        depth: np.ndarray,
        pts: np.ndarray,
        depth_idx: int = 0
    ):
        """
        Z-buffer rendering using lexsort (same as original pipeline).

        Output format matches original: [RGB, D, seg] where D is at depth_idx.
        """
        if len(u) == 0:
            return

        # Sort by (v, u, depth) to get closest point first for each pixel
        sort_idx = np.lexsort((depth, u, v))

        # Find unique (v, u) pairs and their first occurrence (smallest depth)
        vu = np.column_stack((v, u))
        sorted_vu = vu[sort_idx]
        _, unique_idx = np.unique(sorted_vu, axis=0, return_index=True)
        selected_idx = sort_idx[unique_idx]

        # Get pixel coordinates and values
        v_sel = v[selected_idx]
        u_sel = u[selected_idx]
        depth_sel = depth[selected_idx]

        # Fill depth channel at specified index
        pano[v_sel, u_sel, depth_idx] = depth_sel

        # Fill additional channels: [0:depth_idx] and [depth_idx+1:]
        if pts.shape[1] > 3:
            extra_data = pts[selected_idx, 3:]
            n_extra = extra_data.shape[1]

            # Before depth: RGB (0:depth_idx)
            if depth_idx > 0:
                pano[v_sel, u_sel, 0:depth_idx] = extra_data[:, 0:depth_idx]

            # After depth: seg (depth_idx+1:)
            if n_extra > depth_idx:
                pano[v_sel, u_sel, depth_idx+1:] = extra_data[:, depth_idx:]

    def to_5_channel(self, pano_11ch: np.ndarray) -> np.ndarray:
        """
        Convert 11-channel panorama to 5-channel (R, G, B, D, Seg_intensity)

        Args:
            pano_11ch: (H, W, 11) panorama [R, G, B, D, seg*7]

        Returns:
            (H, W, 5) panorama
        """
        pano_5ch = np.zeros((self.height, self.width, 5), dtype=np.float32)

        # RGB, D (format: [R, G, B, D, seg*7])
        pano_5ch[:, :, :4] = pano_11ch[:, :, :4]  # R, G, B, D

        # Seg: argmax of 7 channels -> intensity (0-7)
        seg_7ch = pano_11ch[:, :, 4:11]
        seg_sum = np.sum(seg_7ch, axis=-1)
        seg_int = np.argmax(seg_7ch, axis=-1)
        seg_int[seg_sum == 0] = 7  # No label -> 7 (other)
        pano_5ch[:, :, 4] = seg_int

        return pano_5ch
