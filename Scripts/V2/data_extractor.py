"""
Data Extractor - Extract data from bag files and organize it
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from config import Config, ROS1_TOPICS, ROS2_TOPICS
from bag_reader import BagReader
from pointcloud import PointCloudProcessor
from panorama import PanoramaRenderer


@dataclass
class DataDict:
    """Data dictionary, stores extracted data"""
    # Timestamp lists
    pano_t: List[float] = field(default_factory=list)
    video_t: List[float] = field(default_factory=list)
    pc_t: List[float] = field(default_factory=list)

    # Frame data lists
    pano_frame: List[np.ndarray] = field(default_factory=list)
    video_frame: List[np.ndarray] = field(default_factory=list)
    pc_frame: List[np.ndarray] = field(default_factory=list)
    depth_frame: List[np.ndarray] = field(default_factory=list)

    # Main data array
    data_array: np.ndarray = None

    def to_dict(self) -> Dict:
        """Convert to regular dictionary"""
        result = {}
        if self.pano_t:
            result['pano_t'] = self.pano_t
        if self.pano_frame:
            result['pano_frame'] = np.array(self.pano_frame, dtype=np.float32)
        if self.video_t:
            result['video_t'] = self.video_t
        if self.video_frame:
            result['video_frame'] = self.video_frame
        if self.pc_t:
            result['pc_t'] = self.pc_t
        if self.pc_frame:
            result['pc_frame'] = self.pc_frame
        if self.depth_frame:
            result['depth_frame'] = self.depth_frame
        if self.data_array is not None:
            result['data_array'] = self.data_array
        return result


class DataExtractor:
    """
    Data Extractor

    Corresponds to the original training_set_generator class
    Extracts data from bag files, computes variance, resamples
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.pc_processor = PointCloudProcessor(self.config)
        self.pano_renderer = PanoramaRenderer(
            width=self.config.pano_width,
            height=self.config.pano_height,
            view_dist=self.config.view_dist
        )

        # Extracted data
        self.data = DataDict()

        # Temporary state
        self._temp_pos = np.zeros(3)
        self._temp_quat = np.array([0, 0, 0, 1], dtype=float)
        self._temp_vel = np.zeros(3)
        self._temp_ang_vel = np.zeros(3)
        self._temp_joint = np.zeros(4)
        self._temp_step = 0

        # Indices
        self._pc_idx = 0
        self._video_idx = 0
        self._pano_idx = 0

        # Data row list
        self._data_lines: List[List[float]] = []

        # Point cloud global coordinates
        self._pc_frame_glob: List[np.ndarray] = []

    def extract_from_bag(
        self,
        bag_reader: BagReader,
        save_video: bool = False,
        progress: bool = True
    ):
        """
        Extract data from bag

        Args:
            bag_reader: BagReader instance
            save_video: Whether to save video frames
            progress: Whether to show progress bar
        """
        # Detect format, select topics
        topics = ROS1_TOPICS if bag_reader.format == 'ros1' else ROS2_TOPICS

        total = bag_reader.get_message_count()
        iterator = bag_reader.read_messages()

        if progress:
            iterator = tqdm(iterator, total=total, desc="Extracting data")

        for topic, msg, timestamp in iterator:
            self._extract_message(topic, msg, timestamp, topics, save_video)

        # Convert to numpy array
        if self._data_lines:
            self.data.data_array = np.array(self._data_lines, dtype=float)

    def _extract_message(
        self,
        topic: str,
        msg: Any,
        timestamp: float,
        topics: Dict,
        save_video: bool
    ):
        """Extract single message"""
        new_entry = False

        # Odometry / Pose
        if topic == topics.get('odom'):
            self._extract_odom(msg, timestamp)
            new_entry = True

        # Point Cloud (ROS1 only - /save_pc)
        elif topic == topics.get('pointcloud'):
            self._extract_pointcloud(msg, timestamp)
            new_entry = True

        # Panorama (ROS1 only - /testpano)
        elif topic == topics.get('panorama'):
            self._extract_panorama(msg, timestamp)
            new_entry = True

        # Color image
        elif topic == topics.get('color'):
            if save_video:
                self._extract_video(msg, timestamp)
                new_entry = True

        # Depth image (ROS2)
        elif topic == topics.get('depth'):
            self._extract_depth(msg, timestamp)
            new_entry = True

        # Trigger (ROS2 - /vm_trigger)
        elif topic == topics.get('trigger'):
            self._extract_trigger(msg, timestamp)
            new_entry = True

        # Step counter
        elif topic == topics.get('step'):
            if hasattr(msg, 'data') and msg.data != self._temp_step:
                self._temp_step = msg.data
                new_entry = True

        # Joint states
        elif topic == topics.get('joint'):
            if hasattr(msg, 'position'):
                self._temp_joint = np.array(list(msg.position)[:4])
                new_entry = True

        # Record data row
        if new_entry:
            self._add_data_line(timestamp)

    def _extract_odom(self, msg: Any, timestamp: float):
        """Extract odometry data"""
        # ROS1: nav_msgs/Odometry
        # ROS2: geometry_msgs/PoseStamped or nav_msgs/Odometry
        if hasattr(msg, 'pose'):
            if hasattr(msg.pose, 'pose'):
                # nav_msgs/Odometry
                pose = msg.pose.pose
            else:
                # geometry_msgs/PoseStamped
                pose = msg.pose

            self._temp_pos = np.array([
                pose.position.x,
                pose.position.y,
                pose.position.z
            ])
            self._temp_quat = np.array([
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w
            ])

        if hasattr(msg, 'twist'):
            if hasattr(msg.twist, 'twist'):
                twist = msg.twist.twist
            else:
                twist = msg.twist
            self._temp_vel = np.array([
                twist.linear.x,
                twist.linear.y,
                twist.linear.z
            ])
            self._temp_ang_vel = np.array([
                twist.angular.x,
                twist.angular.y,
                twist.angular.z
            ])

    def _extract_pointcloud(self, msg: Any, timestamp: float):
        """Extract point cloud (ROS1)"""
        try:
            import ros_numpy as rnp
            from depth_processor import fixrgb
        except ImportError:
            print("Warning: ros_numpy not available, skipping pointcloud")
            return

        # Convert point cloud
        pc_data = rnp.point_cloud2.pointcloud2_to_xyz_array(msg, remove_nans=True)
        pc_data = pc_data.reshape(-1, 3)

        # Filter zero points
        valid = np.sum(pc_data, axis=1) > 0.3
        pc_data = pc_data[valid]

        # Apply calibration factor
        pc_data = pc_data * self.config.calib_fac

        # Distance filtering
        dist_sq = np.sum(pc_data**2, axis=1)
        pc_data = pc_data[dist_sq < self.config.view_dist**2]

        self.data.pc_t.append(timestamp)
        self.data.pc_frame.append(pc_data)
        self._pc_idx = len(self.data.pc_t) - 1

    def _extract_panorama(self, msg: Any, timestamp: float):
        """Extract panorama (ROS1)"""
        try:
            import ros_numpy as rnp
            import sensor_msgs
            msg.__class__ = sensor_msgs.msg._Image.Image
            img = rnp.numpify(msg)

            self.data.pano_t.append(timestamp)
            self.data.pano_frame.append(img)
            self._pano_idx = len(self.data.pano_t) - 1
        except Exception as e:
            print(f"Warning: Failed to extract panorama: {e}")

    def _extract_video(self, msg: Any, timestamp: float):
        """Extract video frame"""
        try:
            import ros_numpy as rnp
            import sensor_msgs
            msg.__class__ = sensor_msgs.msg._Image.Image
            img = rnp.numpify(msg)

            self.data.video_t.append(timestamp)
            self.data.video_frame.append(img)
            self._video_idx = len(self.data.video_t) - 1
        except Exception as e:
            print(f"Warning: Failed to extract video: {e}")

    def _extract_depth(self, msg: Any, timestamp: float):
        """Extract depth image (ROS2)"""
        try:
            # ROS2 message format
            if hasattr(msg, 'data'):
                # sensor_msgs/Image
                H, W = msg.height, msg.width
                encoding = msg.encoding

                if encoding == '16UC1':
                    # 16-bit depth in mm
                    depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(H, W)
                    depth = depth.astype(np.float32) / 1000.0  # mm to m
                elif encoding == '32FC1':
                    depth = np.frombuffer(msg.data, dtype=np.float32).reshape(H, W)
                else:
                    print(f"Warning: Unknown depth encoding: {encoding}")
                    return

                self.data.depth_frame.append(depth)
        except Exception as e:
            print(f"Warning: Failed to extract depth: {e}")

    def _extract_trigger(self, msg: Any, timestamp: float):
        """Extract trigger timestamp (ROS2)"""
        # /vm_trigger is std_msgs/Header, records panorama generation moment
        self.data.pano_t.append(timestamp)
        self._pano_idx = len(self.data.pano_t) - 1

    def _add_data_line(self, timestamp: float):
        """
        Add data row

        data_line format (25 columns):
        [0]: time
        [1:4]: position (3)
        [4:8]: quaternion (4)
        [8:11]: pose_variance (3) - filled later
        [11:14]: velocity (3)
        [14:17]: angular_velocity (3)
        [17]: step_count / gait_freq
        [18:22]: joint_angles (4)
        [22]: pc_idx
        [23]: pano_idx
        [24]: video_idx
        """
        data_line = [timestamp]
        data_line.extend(self._temp_pos.tolist())          # 1:4
        data_line.extend(self._temp_quat.tolist())         # 4:8
        data_line.extend([0, 0, 0])                        # 8:11 variance (placeholder)
        data_line.extend(self._temp_vel.tolist())          # 11:14
        data_line.extend(self._temp_ang_vel.tolist())      # 14:17
        data_line.append(self._temp_step)                  # 17
        data_line.extend(self._temp_joint.tolist())        # 18:22
        data_line.extend([self._pc_idx, self._pano_idx, self._video_idx])  # 22:25

        # Avoid duplicates
        if self._data_lines and data_line[1:] == self._data_lines[-1][1:]:
            return

        self._data_lines.append(data_line)

    def compute_pose_variance(self, window_sz: int = 1000):
        """
        Compute pose variance

        Corresponds to original generate_var_pose()
        Uses sliding window to compute position and orientation covariance

        Args:
            window_sz: Sliding window size (number of samples)
        """
        if self.data.data_array is None or len(self.data.data_array) < 3:
            print("Warning: Not enough data to compute variance")
            return

        print("Computing pose variance...")
        for i in tqdm(range(2, len(self.data.data_array))):
            window = self.data.data_array[max(0, i - window_sz):i, 1:8]
            pos, quat = window[:, 0:3], window[:, 3:]

            # Compute direction vectors
            r = R.from_quat(quat)
            r2torso = R.from_rotvec([0, -0.355, 0])
            r_torso = r * r2torso
            rot_array = r_torso.as_matrix()

            ori_vec_x = rot_array[:, [2], 1]
            ori_vec_y = rot_array[:, [0], 2]
            ori_vec_z = -rot_array[:, [0], 1]

            # Combine position and orientation
            rotvec_window = np.hstack([pos, ori_vec_x, ori_vec_y, ori_vec_z])

            # Compute covariance
            cov = np.cov(rotvec_window, rowvar=False).T
            pose_var = np.array([cov[3, 3], cov[4, 4], cov[5, 5]])

            self.data.data_array[i, 8:11] = pose_var

    def resample(self, hz: int = 20):
        """
        Resample data to fixed frequency

        Args:
            hz: Target frequency (Hz)
        """
        if self.data.data_array is None or len(self.data.data_array) < 2:
            print("Warning: Not enough data to resample")
            return

        print(f"Resampling to {hz} Hz...")
        data_t = self.data.data_array[:, 0]
        t_start = data_t[0]
        t_end = data_t[-1]
        t_step = 1.0 / hz

        resampled = []
        for t in tqdm(np.arange(t_start, t_end - 1, t_step)):
            idx = np.argmin(np.abs(data_t - t))
            line = [t] + list(self.data.data_array[idx, 1:])
            resampled.append(line)

        self.data.data_array = np.array(resampled, dtype=float)
        print(f"Resampled to {len(resampled)} samples")

    def compute_gait_freq(self, window_sec: float = 2.5, hz: int = 20):
        """
        Compute gait frequency

        Uses zero-crossing of joint angles to calculate

        Args:
            window_sec: Window duration (seconds)
            hz: Data frequency
        """
        if self.data.data_array is None or len(self.data.data_array) < 10:
            return

        print("Computing gait frequency...")
        joint_angles = self.data.data_array[:, 18]  # Take first joint angle
        window_sz = int(window_sec * hz)

        # Compute zero crossings
        joint_mean = 0
        counts = np.zeros(len(joint_angles))

        for i in range(len(joint_angles) - 1):
            joint_mean = (joint_mean * i + joint_angles[i]) / (i + 1)
            if joint_angles[i] < joint_mean < joint_angles[i + 1]:
                counts[i] += 1
            if joint_angles[i + 1] < joint_mean < joint_angles[i]:
                counts[i] += 1

        # Compute frequency
        for i in range(len(counts)):
            freq = np.sum(counts[max(0, i - window_sz):i + 1]) / (window_sz / hz)
            self.data.data_array[i, 17] = freq

    def convert_pc_to_global(self):
        """
        Convert all point clouds to global coordinate frame
        """
        if not self.data.pc_frame:
            return

        print("Converting point clouds to global frame...")
        self._pc_frame_glob = []

        for i, (pc, t) in enumerate(tqdm(zip(self.data.pc_frame, self.data.pc_t))):
            pos, quat = self.find_nearest_pose(t)

            # Camera -> Body -> World
            pc_world = self.pc_processor.camera_to_world(pc, pos, quat)
            self._pc_frame_glob.append(pc_world)

    def generate_panoramas(self, window_sz: int = None):
        """
        Generate panoramas (from point clouds)

        Args:
            window_sz: Point cloud window size (number of frames)
        """
        if window_sz is None:
            window_sz = self.config.window_sz

        if not self._pc_frame_glob:
            self.convert_pc_to_global()

        if not self._pc_frame_glob:
            print("Warning: No point clouds for panorama generation")
            return

        print("Generating panoramas...")
        self.data.pano_frame = []

        for i, t in enumerate(tqdm(self.data.pano_t)):
            pos, quat = self.find_nearest_pose(t)

            # Get yaw matrix (horizon lock)
            euler = R.from_quat(quat).as_euler('ZYX')
            yaw_matrix = R.from_euler('ZYX', [euler[0], 0, 0]).as_matrix()

            # Select point clouds within window
            pc_idx = np.sum(np.array(self.data.pc_t) < t)
            window_l = max(0, pc_idx - window_sz + 1)
            window_r = pc_idx + 1

            pc_window = self._pc_frame_glob[window_l:window_r]

            # Render panorama
            pano = self.pano_renderer.render_depth_only(
                pc_window, pos, yaw_matrix, filter_dist=True
            )
            self.data.pano_frame.append(pano)

    def find_nearest_pose(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find nearest pose

        Args:
            t: Query timestamp

        Returns:
            (position, quaternion)
        """
        if self.data.data_array is None:
            return np.zeros(3), np.array([0, 0, 0, 1])

        idx = np.argmin(np.abs(self.data.data_array[:, 0] - t))
        pos = self.data.data_array[idx, 1:4].copy()
        quat = self.data.data_array[idx, 4:8].copy()
        return pos, quat

    def get_data_dict(self) -> Dict:
        """Get data dictionary"""
        return self.data.to_dict()
