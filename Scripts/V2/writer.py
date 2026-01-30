"""
Training set output - Save in joblib format
"""

import os
import time
import numpy as np
from typing import Dict, Optional
from datetime import datetime


class TrainingSetWriter:
    """
    Training set writer

    Saves processed data in joblib format
    """

    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: Output directory
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save(
        self,
        data_dict: Dict,
        prefix: str = 'eDS20HZVZS',
        bag_name: str = None,
        compress: bool = True
    ) -> str:
        """
        Save training set

        Args:
            data_dict: Data dictionary
            prefix: Filename prefix
            bag_name: bag filename (used to generate output filename)
            compress: Whether to compress

        Returns:
            Saved file path
        """
        import joblib

        # Generate filename
        if bag_name:
            # Remove extension
            base_name = os.path.splitext(os.path.basename(bag_name))[0]
            filename = f"{prefix}_{base_name}"
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{prefix}_{timestamp}"

        filepath = os.path.join(self.output_dir, filename)

        print(f"Saving to {filepath}...")
        t_start = time.time()

        # Save
        if compress:
            joblib.dump(data_dict, filepath, compress=('lz4', 1))
        else:
            joblib.dump(data_dict, filepath)

        t_end = time.time()
        print(f"Done saving, took {t_end - t_start:.2f}s")

        return filepath

    def save_minimal(
        self,
        data_dict: Dict,
        prefix: str = 'eDS20HZVZS',
        bag_name: str = None
    ) -> str:
        """
        Save minimal training set (remove unnecessary data)

        Only keeps:
        - pano_t
        - pano_frame (5ch: RGB+D+Seg)
        - data_array

        Args:
            data_dict: Data dictionary
            prefix: Filename prefix
            bag_name: bag filename

        Returns:
            Saved file path
        """
        # Create minimal dictionary
        minimal_dict = {}

        if 'pano_t' in data_dict:
            minimal_dict['pano_t'] = data_dict['pano_t']

        if 'pano_frame' in data_dict:
            minimal_dict['pano_frame'] = data_dict['pano_frame']

        if 'data_array' in data_dict:
            minimal_dict['data_array'] = data_dict['data_array']

        return self.save(minimal_dict, prefix, bag_name, compress=True)


def convert_pano_to_5ch(pano_11ch: np.ndarray) -> np.ndarray:
    """
    Convert 11-channel panorama to 5-channel

    Input: (N, H, W, 11) - RGB(3) + D(1) + Seg_onehot(7)
    Output: (N, 5, H, W) - Note the dimension order change!

    Original training data format:
    - pano shape: (N, 5, 140, 360)
    - channels: R, G, B, D, Seg_intensity (0-7)
    """
    N, H, W, C = pano_11ch.shape
    assert C == 11, f"Expected 11 channels, got {C}"

    # RGB + D
    pano_5ch = np.zeros((N, 5, H, W), dtype=np.float32)
    pano_5ch[:, 0, :, :] = pano_11ch[:, :, :, 0]  # R
    pano_5ch[:, 1, :, :] = pano_11ch[:, :, :, 1]  # G
    pano_5ch[:, 2, :, :] = pano_11ch[:, :, :, 2]  # B
    pano_5ch[:, 3, :, :] = pano_11ch[:, :, :, 3]  # D

    # Seg: argmax of 7 channels
    seg_7ch = pano_11ch[:, :, :, 4:11]
    seg_sum = np.sum(seg_7ch, axis=-1)
    seg_int = np.argmax(seg_7ch, axis=-1)
    seg_int[seg_sum == 0] = 7  # No label → other
    pano_5ch[:, 4, :, :] = seg_int

    return pano_5ch


def crop_pano_height(pano: np.ndarray, target_height: int = 140) -> np.ndarray:
    """
    Crop panorama height

    Crop from 180 to 140 (remove top and bottom 20 pixels each)

    Args:
        pano: (N, C, H, W) or (H, W, C) panorama
        target_height: Target height

    Returns:
        Cropped panorama
    """
    if pano.ndim == 4:
        # (N, C, H, W)
        H = pano.shape[2]
        if H <= target_height:
            return pano
        crop = (H - target_height) // 2
        return pano[:, :, crop:crop + target_height, :]
    elif pano.ndim == 3:
        # (H, W, C)
        H = pano.shape[0]
        if H <= target_height:
            return pano
        crop = (H - target_height) // 2
        return pano[crop:crop + target_height, :, :]
    else:
        raise ValueError(f"Unexpected pano shape: {pano.shape}")
