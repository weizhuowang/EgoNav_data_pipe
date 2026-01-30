"""
Data Pipeline V2 - Modular data processing pipeline

Supports ROS1 bag and ROS2 mcap formats, generates training datasets.
"""

from config import Config, D455_INTRINSICS, PANO_WIDTH, PANO_HEIGHT
from bag_reader import BagReader
from pointcloud import PointCloudProcessor
from panorama import PanoramaRenderer
from depth_processor import DepthProcessor
from data_extractor import DataExtractor
from labeler import SemanticLabeler
from writer import TrainingSetWriter

__all__ = [
    'Config',
    'D455_INTRINSICS',
    'PANO_WIDTH',
    'PANO_HEIGHT',
    'BagReader',
    'PointCloudProcessor',
    'PanoramaRenderer',
    'DepthProcessor',
    'DataExtractor',
    'SemanticLabeler',
    'TrainingSetWriter',
]
