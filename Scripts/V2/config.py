"""
Configuration - Camera intrinsics, panorama parameters, semantic classes
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class CameraIntrinsics:
    """Camera intrinsics"""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


# D455 depth camera intrinsics
D455_INTRINSICS = CameraIntrinsics(
    fx=431.0087890625,
    fy=431.0087890625,
    cx=429.328704833984,
    cy=242.162155151367,
    width=848,
    height=480
)

# Panorama parameters
PANO_WIDTH = 360
PANO_HEIGHT = 180
PANO_U_STEP = 1.0
PANO_V_STEP = 1.0

# Semantic class definitions (8 classes)
# 0: floor/ground
# 1: wall
# 2: stair
# 3: door
# 4: obstacle (furniture, plants, etc.)
# 5: human
# 6: terrain (grass, sand, etc.)
# 7: other/unknown
SEMANTIC_CLASSES = [
    'floor', 'wall', 'stair', 'door', 'obstacle', 'human', 'terrain', 'other'
]
NUM_CLASSES = len(SEMANTIC_CLASSES)

# ADE20K -> 8 class mapping table (ade2our)
# ADE20K has 150 classes, index 0 is background, 1-150 are categories
# Index i maps ADE20K class i-1 to our target class
ADE2OUR = np.array([
    7,  # 0: background -> other
    3,  # 1: wall -> wall (idx 1 in our classes, but we use 3 for wall)
    3,  # 2: building -> wall
    7,  # 3: sky -> other
    0,  # 4: floor -> floor
    4,  # 5: tree -> obstacle
    7,  # 6: ceiling -> other
    0,  # 7: road -> floor
    4,  # 8: bed -> obstacle
    4,  # 9: windowpane -> obstacle
    6,  # 10: grass -> terrain
    4,  # 11: cabinet -> obstacle
    0,  # 12: sidewalk -> floor
    5,  # 13: person -> human
    6,  # 14: earth -> terrain
    2,  # 15: door -> stair (originally door, but ade20k mapping)
    4,  # 16: table -> obstacle
    7,  # 17: mountain -> other
    4,  # 18: plant -> obstacle
    4,  # 19: curtain -> obstacle
    4,  # 20: chair -> obstacle
    5,  # 21: car -> human (dynamic object)
    4,  # 22: water -> obstacle
    4,  # 23: painting -> obstacle
    4,  # 24: sofa -> obstacle
    4,  # 25: shelf -> obstacle
    3,  # 26: house -> wall
    4,  # 27: sea -> obstacle
    3,  # 28: mirror -> wall
    0,  # 29: rug -> floor
    6,  # 30: field -> terrain
    4,  # 31: armchair -> obstacle
    4,  # 32: seat -> obstacle
    3,  # 33: fence -> wall
    4,  # 34: desk -> obstacle
    4,  # 35: rock -> obstacle
    4,  # 36: wardrobe -> obstacle
    4,  # 37: lamp -> obstacle
    4,  # 38: bathtub -> obstacle
    4,  # 39: railing -> obstacle
    4,  # 40: cushion -> obstacle
    4,  # 41: base -> obstacle
    4,  # 42: box -> obstacle
    3,  # 43: column -> wall
    4,  # 44: signboard -> obstacle
    4,  # 45: chest -> obstacle
    4,  # 46: counter -> obstacle
    6,  # 47: sand -> terrain
    4,  # 48: sink -> obstacle
    3,  # 49: skyscraper -> wall
    4,  # 50: fireplace -> obstacle
    4,  # 51: refrigerator -> obstacle
    7,  # 52: grandstand -> other
    0,  # 53: path -> floor
    2,  # 54: stairs -> stair
    0,  # 55: runway -> floor
    4,  # 56: case -> obstacle
    4,  # 57: pool table -> obstacle
    4,  # 58: pillow -> obstacle
    2,  # 59: screen door -> stair
    2,  # 60: stairway -> stair
    4,  # 61: river -> obstacle
    3,  # 62: bridge -> wall
    4,  # 63: bookcase -> obstacle
    4,  # 64: blind -> obstacle
    4,  # 65: coffee table -> obstacle
    4,  # 66: toilet -> obstacle
    4,  # 67: flower -> obstacle
    4,  # 68: book -> obstacle
    6,  # 69: hill -> terrain
    4,  # 70: bench -> obstacle
    4,  # 71: countertop -> obstacle
    4,  # 72: stove -> obstacle
    4,  # 73: palm -> obstacle
    4,  # 74: kitchen island -> obstacle
    4,  # 75: computer -> obstacle
    4,  # 76: swivel chair -> obstacle
    5,  # 77: boat -> human
    4,  # 78: bar -> obstacle
    4,  # 79: arcade machine -> obstacle
    3,  # 80: hovel -> wall
    5,  # 81: bus -> human
    4,  # 82: towel -> obstacle
    7,  # 83: light -> other
    5,  # 84: truck -> human
    3,  # 85: tower -> wall
    4,  # 86: chandelier -> obstacle
    7,  # 87: awning -> other
    4,  # 88: streetlight -> obstacle
    3,  # 89: booth -> wall
    4,  # 90: television -> obstacle
    5,  # 91: airplane -> human
    0,  # 92: dirt track -> floor
    4,  # 93: apparel -> obstacle
    4,  # 94: pole -> obstacle
    0,  # 95: land -> floor
    4,  # 96: bannister -> obstacle
    2,  # 97: escalator -> stair
    4,  # 98: ottoman -> obstacle
    4,  # 99: bottle -> obstacle
    4,  # 100: buffet -> obstacle
    3,  # 101: poster -> wall
    0,  # 102: stage -> floor
    5,  # 103: van -> human
    5,  # 104: ship -> human
    4,  # 105: fountain -> obstacle
    4,  # 106: conveyer belt -> obstacle
    7,  # 107: canopy -> other
    4,  # 108: washer -> obstacle
    4,  # 109: plaything -> obstacle
    4,  # 110: swimming pool -> obstacle
    4,  # 111: stool -> obstacle
    4,  # 112: barrel -> obstacle
    4,  # 113: basket -> obstacle
    6,  # 114: waterfall -> terrain
    4,  # 115: tent -> obstacle
    4,  # 116: bag -> obstacle
    5,  # 117: minibike -> human
    4,  # 118: cradle -> obstacle
    4,  # 119: oven -> obstacle
    4,  # 120: ball -> obstacle
    4,  # 121: food -> obstacle
    2,  # 122: step -> stair
    4,  # 123: tank -> obstacle
    4,  # 124: trade name -> obstacle
    4,  # 125: microwave -> obstacle
    4,  # 126: pot -> obstacle
    5,  # 127: animal -> human
    5,  # 128: bicycle -> human
    4,  # 129: lake -> obstacle
    4,  # 130: dishwasher -> obstacle
    4,  # 131: screen -> obstacle
    4,  # 132: blanket -> obstacle
    4,  # 133: sculpture -> obstacle
    4,  # 134: hood -> obstacle
    4,  # 135: sconce -> obstacle
    4,  # 136: vase -> obstacle
    4,  # 137: traffic light -> obstacle
    4,  # 138: tray -> obstacle
    4,  # 139: ashcan -> obstacle
    4,  # 140: fan -> obstacle
    4,  # 141: pier -> obstacle
    4,  # 142: crt screen -> obstacle
    4,  # 143: plate -> obstacle
    4,  # 144: monitor -> obstacle
    4,  # 145: bulletin board -> obstacle
    7,  # 146: shower -> other
    4,  # 147: radiator -> obstacle
    4,  # 148: glass -> obstacle
    4,  # 149: clock -> obstacle
    4,  # 150: flag -> obstacle
], dtype=np.uint8)

# ROS1 Topics
ROS1_TOPICS = {
    'odom': '/t265/odom/sample',
    'pointcloud': '/save_pc',
    'panorama': '/testpano',
    'color': '/d400/color/image_raw',
    'depth': '/d400/depth/image_raw',
    'step': '/step_counter',
    'joint': '/joint_states',
}

# ROS2 Topics (for mcap)
ROS2_TOPICS = {
    'odom': '/T265/pose/sample',
    'color': '/D400/color/image_raw',
    'depth': '/D400/depth/image_rect_raw',
    'trigger': '/vm_trigger',
}


@dataclass
class Config:
    """Data processing configuration"""
    intrinsics: CameraIntrinsics = None
    pano_width: int = PANO_WIDTH
    pano_height: int = PANO_HEIGHT
    view_dist: float = 10.0
    window_sz: int = 32
    resample_hz: int = 20
    calib_fac: float = 1.0
    sample_ratio: float = 0.3

    def __post_init__(self):
        if self.intrinsics is None:
            self.intrinsics = D455_INTRINSICS
