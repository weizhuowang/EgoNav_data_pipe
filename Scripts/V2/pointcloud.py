"""
Point cloud processing - coordinate transforms, depth to point cloud
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple
from config import Config, CameraIntrinsics, D455_INTRINSICS


class PointCloudProcessor:
    """
    Point cloud processor

    Responsible for:
    - Depth to pointcloud
    - Camera to body to world frame transforms
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.intrinsics = self.config.intrinsics

    def depth_to_pointcloud(
        self,
        depth: np.ndarray,
        color: np.ndarray = None,
        seg: np.ndarray = None,
        min_depth: float = 0.3
    ) -> np.ndarray:
        """
        Depth image to pointcloud

        Args:
            depth: (H, W) or (H, W, 1) depth map, in meters
            color: (H, W, 3) RGB image, optional
            seg: (H, W, C) segmentation map, optional
            min_depth: minimum valid depth

        Returns:
            pointcloud array, shape (N, 3+3+C) - xyz + rgb + seg
        """
        # Ensure depth is 2D
        if depth.ndim == 3:
            depth = depth[:, :, 0]

        H, W = depth.shape
        fx, fy = self.intrinsics.fx, self.intrinsics.fy
        cx, cy = self.intrinsics.cx, self.intrinsics.cy

        # Create pixel coordinate grid
        v, u = np.indices((H, W))

        # Valid depth mask
        mask = depth > min_depth

        # Extract valid points
        u_valid = u[mask]
        v_valid = v[mask]
        z_valid = depth[mask]

        # Back-project to 3D
        x = (u_valid - cx) * z_valid / fx
        y = (v_valid - cy) * z_valid / fy
        z = z_valid

        # Combine xyz
        points = np.column_stack([x, y, z])

        # Add color
        if color is not None:
            rgb = color[v_valid, u_valid, :3]
            points = np.column_stack([points, rgb])
        else:
            # Fill with zeros
            points = np.column_stack([points, np.zeros((len(x), 3))])

        # Add segmentation
        if seg is not None:
            seg_vals = seg[v_valid, u_valid]
            if seg_vals.ndim == 1:
                seg_vals = seg_vals[:, np.newaxis]
            points = np.column_stack([points, seg_vals])

        return points.astype(np.float32)

    def camera_to_body(self, pc: np.ndarray) -> np.ndarray:
        """
        Camera frame to body frame

        Original code transform:
        pt_array[:,[0,1,2]] = pt_array[:,[2,0,1]]  # xyz reorder
        pt_array[:,1] = -pt_array[:,1]             # negate y
        pt_array[:,2] = -pt_array[:,2]             # negate z

        i.e.: x_body = z_cam, y_body = -x_cam, z_body = -y_cam
        """
        pc_body = pc.copy()
        # xyz reorder: [x,y,z] -> [z,x,y]
        xyz = pc_body[:, :3].copy()
        pc_body[:, 0] = xyz[:, 2]   # x_body = z_cam
        pc_body[:, 1] = -xyz[:, 0]  # y_body = -x_cam
        pc_body[:, 2] = -xyz[:, 1]  # z_body = -y_cam
        return pc_body

    def body_to_world(
        self,
        pc: np.ndarray,
        position: np.ndarray,
        quaternion: np.ndarray
    ) -> np.ndarray:
        """
        Body frame to world frame

        Args:
            pc: (N, 3+) pointcloud, first 3 columns are xyz
            position: (3,) position [x, y, z]
            quaternion: (4,) quaternion [x, y, z, w]

        Returns:
            pointcloud in world coordinate system
        """
        pc_world = pc.copy()
        rot = R.from_quat(quaternion).as_matrix()

        # Rotation + translation
        xyz_body = pc_world[:, :3]
        xyz_world = (rot @ xyz_body.T).T + position

        pc_world[:, :3] = xyz_world
        return pc_world

    def camera_to_world(
        self,
        pc: np.ndarray,
        position: np.ndarray,
        quaternion: np.ndarray
    ) -> np.ndarray:
        """
        Camera to world frame (combined transform)
        """
        pc_body = self.camera_to_body(pc)
        pc_world = self.body_to_world(pc_body, position, quaternion)
        return pc_world

    def transform_to_local(
        self,
        pc_world: np.ndarray,
        position: np.ndarray,
        yaw_matrix: np.ndarray
    ) -> np.ndarray:
        """
        World coordinate system -> local coordinate system (with current position as origin, yaw as heading)

        For panorama generation, only considers yaw rotation

        Args:
            pc_world: (N, 3+) world coordinate pointcloud
            position: (3,) current position
            yaw_matrix: (3, 3) yaw rotation matrix
        """
        pc_local = pc_world.copy()
        xyz = pc_local[:, :3] - position  # Translate to origin
        xyz = xyz @ yaw_matrix  # Rotate to local frame
        pc_local[:, :3] = xyz
        return pc_local

    @staticmethod
    def get_yaw_matrix(quaternion: np.ndarray) -> np.ndarray:
        """
        Extract yaw angle rotation matrix from quaternion (GoPro-like horizon lock)

        Keep only Z-axis rotation, ignore pitch and roll
        """
        euler = R.from_quat(quaternion).as_euler('ZYX')
        yaw = euler[0]  # Z-axis rotation angle
        return R.from_euler('ZYX', [yaw, 0, 0]).as_matrix()

    def filter_by_distance(
        self,
        pc: np.ndarray,
        max_dist: float = 10.0,
        min_dist: float = 0.0
    ) -> np.ndarray:
        """
        Filter pointcloud by distance

        Args:
            pc: pointcloud (N, 3+)
            max_dist: max distance
            min_dist: min distance

        Returns:
            filtered pointcloud
        """
        xyz = pc[:, :3]
        dist = np.linalg.norm(xyz, axis=1)
        mask = (dist >= min_dist) & (dist <= max_dist)
        return pc[mask]

    def subsample(
        self,
        pc: np.ndarray,
        ratio: float = 0.3
    ) -> np.ndarray:
        """
        Randomly subsample pointcloud

        Args:
            pc: pointcloud (N, M)
            ratio: Keep ratio (0, 1]

        Returns:
            subsampled pointcloud
        """
        n_points = pc.shape[0]
        n_keep = int(n_points * ratio)
        if n_keep >= n_points:
            return pc
        indices = np.random.choice(n_points, n_keep, replace=False)
        return pc[indices]
