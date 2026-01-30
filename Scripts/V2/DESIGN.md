# Data Pipeline V2 设计计划

## 目标
创建模块化的 V2 数据管道，整合现有分散的 ROS1 pipeline，支持 ROS1 bag 和 ROS2 mcap。

## 原始 Pipeline 流程理解

```
lag_compensate.py → read_bagV3.py → add_video_frame.py → DINOv2_labeler.py
      ↓                   ↓                  ↓                   ↓
  修复时间戳          提取数据+生成pano    添加video/depth     语义分割+重建pano
```

### 1. lag_compensate.py
- 将 bag time 改为 msg.header.stamp (修复录制延迟)

### 2. read_bagV3.py (training_set_generator)
- 提取 topics: `/t265/odom/sample`, `/save_pc`, `/testpano`, `/d400/color/image_raw`
- 点云坐标转换: `pc2globalpc()` - camera→body→world
  ```python
  pt_array[:,[0,1,2]] = pt_array[:,[2,0,1]]  # xyz重排
  pt_array[:,1] = -pt_array[:,1]             # y取反
  pt_array[:,2] = -pt_array[:,2]             # z取反
  global_xyz = r.dot(pt_array[:,:3].T).T + curr_pos
  ```
- 全景生成: `generate_pano()` - 360x180 柱面投影，z-buffer
- 姿态方差: `generate_var_pose()` - 滑动窗口计算
- 重采样: 20Hz
- 步态频率: `update_gait()` - IMU关节角度计算

### 3. add_video_frame.py (Data_Processor)
- 从 bag 提取 video_frame 和 pc_frame (带 RGB)
- 点云→深度图: `point_to_channelD455()` 使用相机内参重投影
- 边缘移除: `remove_edges()` - Canny + erode

### 4. DINOv2_labeler.py (semantic_labeler)
- DINOv2 ADE20K 分割 → 8类映射 (ade2our)
- 分割帧→点云: `depth_to_pc()` 使用内参
- 点云→全局: `local_pc_to_global()`
- 重建全景: `redo_pano()` - 包含 RGB+D+Seg(7ch)
- 最终: 5通道 (R,G,B,D,Seg-intensity)

## data_array 列定义 (25列)
```
[0]: time
[1:4]: position (x,y,z)
[4:8]: quaternion (x,y,z,w)
[8:11]: pose_variance (3)
[11:14]: velocity (3)
[14:17]: angular_velocity (3)
[17]: gait_freq
[18:22]: joint_angles (4)
[22]: pc_idx
[23]: pano_idx
[24]: video_idx
```

## V2 目录结构
```
egonav_deploy/data_pipeline/
├── __init__.py
├── config.py           # 相机内参、全景参数、语义类别
├── bag_reader.py       # 统一 BagReader (支持 ros1 bag + mcap)
├── data_extractor.py   # 数据提取 (对应 read_bagV3)
├── pointcloud.py       # 点云处理 (坐标转换)
├── panorama.py         # 全景生成 (z-buffer投影)
├── depth_processor.py  # 深度处理 (pc→depth, remove_edges)
├── labeler.py          # 语义分割 (DINOv2)
├── writer.py           # 输出 joblib
└── data_pipe_V2.py     # CLI 主入口
```

## 核心类设计

### 1. BagReader (bag_reader.py)
```python
class BagReader:
    """支持 ROS1 bag 和 ROS2 mcap 的统一接口"""
    def __init__(self, bag_path: str)
    def read_messages(self, topics: List[str] = None) -> Iterator[Tuple[str, Any, float]]
    def get_message_time(self, msg) -> float  # 统一获取 header.stamp
    @staticmethod
    def detect_format(path: str) -> str  # 'ros1' or 'mcap'
```

### 2. DataExtractor (data_extractor.py)
```python
class DataExtractor:
    """对应 training_set_generator，提取和组织数据"""
    def __init__(self, bag_reader: BagReader, config: Config)
    def extract_all(self) -> DataDict
    def find_nearest_pose(self, t: float) -> Tuple[np.ndarray, np.ndarray]
    def resample(self, hz: int = 20) -> None
    def compute_pose_variance(self, window_sz: int = 1000) -> None
    def compute_gait_freq(self, window_sz: float = 2.5) -> None
```

### 3. PointCloudProcessor (pointcloud.py)
```python
class PointCloudProcessor:
    """点云坐标转换"""
    def __init__(self, config: Config)
    def camera_to_body(self, pc: np.ndarray) -> np.ndarray
        # xyz重排 + y,z取反
    def body_to_world(self, pc: np.ndarray, pos: np.ndarray, quat: np.ndarray) -> np.ndarray
    def depth_to_pointcloud(self, depth: np.ndarray, color: np.ndarray = None,
                           seg: np.ndarray = None) -> np.ndarray
```

### 4. PanoramaRenderer (panorama.py)
```python
class PanoramaRenderer:
    """全景图生成 - 对应 generate_pano()"""
    def __init__(self, width: int = 360, height: int = 180, view_dist: float = 10.0)
    def render(self, pc_frames_world: List[np.ndarray], curr_pos: np.ndarray,
               curr_yaw_matrix: np.ndarray) -> np.ndarray
        # 柱面投影 + z-buffer
    def lidar_to_surround_coords(self, x, y, z, dxy) -> Tuple[np.ndarray, np.ndarray]
```

### 5. DepthProcessor (depth_processor.py)
```python
class DepthProcessor:
    """深度图处理"""
    def __init__(self, config: Config)
    def pointcloud_to_depth(self, pc: np.ndarray) -> np.ndarray
        # 对应 point_to_channelD455
    def remove_edges(self, depth: np.ndarray, kernel_size: int = 10) -> np.ndarray
```

### 6. SemanticLabeler (labeler.py)
```python
class SemanticLabeler:
    """DINOv2 语义分割"""
    def __init__(self, device: str = 'cuda')
    def load_model(self) -> None
    def segment(self, rgb_bgr: np.ndarray) -> np.ndarray  # 返回 (H,W,7) one-hot
    def ade20k_to_classes(self, seg_ade: np.ndarray) -> np.ndarray
        # 使用 ade2our 映射表
```

### 7. TrainingSetWriter (writer.py)
```python
class TrainingSetWriter:
    """输出 joblib 格式"""
    def __init__(self, output_dir: str)
    def save(self, data_dict: dict, prefix: str = 'eDS20HZVZS') -> str
```

## CLI 接口 (data_pipe_V2.py)
```bash
python data_pipe_V2.py \
    --bag /path/to/bag_or_mcap \
    --output /path/to/output_dir \
    [--no-seg]          # 跳过语义分割
    [--no-resample]     # 不重采样
    [--hz 20]           # 重采样频率
    [--window-sz 32]    # 全景窗口大小
    [--view-dist 10.0]  # 全景视距
    [--viz]             # 显示处理进度
```

## 关键参数 (config.py)
```python
# D455 内参
D455_INTRINSICS = CameraIntrinsics(
    fx=431.0087890625, fy=431.0087890625,
    cx=429.328704833984, cy=242.162155151367,
    width=848, height=480
)

# 全景参数
PANO_WIDTH = 360
PANO_HEIGHT = 180  # 原始是180，裁剪后可能是140

# 语义类别 (8类)
CLASSES = ['floor', 'wall', 'stair', 'door', 'obstacle', 'human', 'terrain', 'other']

# ADE20K → 8类映射表 (ade2our)
ADE2OUR = np.array([7, 3, 3, 7, 0, 4, ...])  # 150+1 个值
```

## 待创建文件
```
egonav_deploy/data_pipeline/
├── __init__.py
├── config.py
├── bag_reader.py
├── data_extractor.py
├── pointcloud.py
├── panorama.py
├── depth_processor.py
├── labeler.py
├── writer.py
└── data_pipe_V2.py
```

## 验证方法
```bash
# 1. 用原始 ROS1 bag 测试，对比输出
python data_pipe_V2.py --bag test.bag --output test_out_v2

# 2. 对比原始 pipeline 输出
python -c "
import joblib
old = joblib.load('eDS20HZVZS_test')
new = joblib.load('test_out_v2/eDS20HZVZS_test')
print('pano_frame match:', np.allclose(old['pano_frame'], new['pano_frame']))
print('data_array match:', np.allclose(old['data_array'], new['data_array']))
"

# 3. 检查输出格式
# 期望 pano_frame: (N, 180, 360, 5) 或裁剪后 (N, 140, 360, 5)
# 期望 data_array: (M, 25)
```

## 快速运行指南

### 环境
必须在 `GSAM` conda 环境中运行:
```bash
conda activate GSAM
```

### 文件路径
- Bag 文件: `/arm/u/weizhuo2/Documents/Data_pipe/Bags/`
- 输出目录: `/arm/u/weizhuo2/Documents/Data_pipe/Training_sets/V2_test/`
- 原始 eDS 参考: `/arm/u/weizhuo2/Documents/Data_pipe/Training_sets/`

### 运行命令
```bash
cd /arm/u/weizhuo2/Documents/Data_pipe/Scripts/V2

# 完整运行 (带语义分割)
python data_pipe_V2.py \
    --bag /arm/u/weizhuo2/Documents/Data_pipe/Bags/V2DataRedo_field_lag.bag \
    --output /arm/u/weizhuo2/Documents/Data_pipe/Training_sets/V2_test \
    --prefix V2TEST{version_number}
```

### 输出格式
V2 输出应与原始 eDS 完全一致:

### DEBUG 模式
代码中有 两处`DEBUG_LIMIT = 500` 限制全景帧数。
确认输出正确后，注释掉该行以生成完整数据集。

## Bug 修复记录

### 2025-01-30: data_array 长度不匹配
**问题**: V2 的 data_array 长度 (3842) 与原始 eDS (3791) 不一致

**原因**: 原始代码有 `valid_start` 逻辑，在 resample 前剪掉第一个 pc/video 之前的数据

**修复**: 在 `data_extractor.py` 中添加:
1. `_valid_start` 成员变量
2. 在 `_extract_video()` 和 `_extract_pointcloud()` 中更新 `_valid_start`
3. 在 `resample()` 开始时剪掉 `_valid_start` 之前的数据

### 2025-01-30: 全景图稀疏 (RGB mean ~14 vs ~57)
**问题**: V2 全景 RGB 均值只有 14，原始 eDS 是 57

**原因**: `DepthProcessor.pointcloud_to_depth()` 归一化深度到 0-1，但 `SimpleLabelProjector` 直接使用该值做反投影

**修复**: 在 `data_pipe_V2.py` 的 `_run_segmentation_pipeline()` 中:
```python
depth_frame = depth_processor.pointcloud_to_depth(pc)[:, :, 0]
depth_frame = depth_frame * 10.0  # 恢复实际米数
```
