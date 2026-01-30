#!/usr/bin/env python3
"""
Data Pipeline V2 - Modular Data Processing Pipeline

Converts ROS1 bag or ROS2 mcap to training dataset.

Usage:
    python data_pipe_V2.py --bag <bag_path> --output <output_dir>

Examples:
    # Process ROS1 bag
    python data_pipe_V2.py --bag /path/to/test.bag --output ./output

    # Process ROS2 mcap
    python data_pipe_V2.py --bag /path/to/V4Data.mcap --output ./output

    # Skip semantic segmentation (quick test)
    python data_pipe_V2.py --bag test.bag --output ./output --no-seg

    # Show visualization
    python data_pipe_V2.py --bag test.bag --output ./output --viz
"""

import argparse
import os
import sys
import time
import numpy as np
from typing import Dict, Optional

from config import Config
from bag_reader import BagReader
from data_extractor import DataExtractor
from pointcloud import PointCloudProcessor
from panorama import PanoramaRenderer
from depth_processor import DepthProcessor
from labeler import SemanticLabeler, SimpleLabelProjector
from writer import TrainingSetWriter, convert_pano_to_5ch, crop_pano_height


def process_bag(
    bag_path: str,
    output_dir: str,
    config: Config = None,
    do_seg: bool = True,
    do_resample: bool = True,
    viz: bool = False,
    save_video: bool = False
) -> str:
    """
    Process bag file and generate training dataset

    Args:
        bag_path: Path to bag file
        output_dir: Output directory
        config: Configuration object
        do_seg: Whether to perform semantic segmentation
        do_resample: Whether to resample
        viz: Whether to show visualization
        save_video: Whether to save video frames

    Returns:
        Path to saved file
    """
    if config is None:
        config = Config()

    print("=" * 60)
    print(f"Processing: {bag_path}")
    print("=" * 60)

    t_total_start = time.time()

    # 1. Read bag
    print("\n[1/6] Opening bag file...")
    reader = BagReader(bag_path)
    print(f"Format: {reader.format}")
    print(f"Messages: {reader.get_message_count()}")

    # 2. Extract data
    print("\n[2/6] Extracting data...")
    extractor = DataExtractor(config)
    extractor.extract_from_bag(reader, save_video=save_video, progress=True)
    reader.close()

    data_dict = extractor.get_data_dict()
    print(f"Data array shape: {data_dict.get('data_array', np.array([])).shape}")
    print(f"Point clouds: {len(data_dict.get('pc_t', []))}")
    print(f"Panorama timestamps: {len(data_dict.get('pano_t', []))}")

    # 3. Compute variance
    print("\n[3/6] Computing pose variance...")
    extractor.compute_pose_variance(window_sz=1000)

    # 4. Resample
    if do_resample:
        print(f"\n[4/6] Resampling to {config.resample_hz} Hz...")
        extractor.resample(hz=config.resample_hz)

    # 5. Gait frequency
    print("\n[5/6] Computing gait frequency...")
    extractor.compute_gait_freq(window_sec=2.5, hz=config.resample_hz)

    # 6. Panorama generation / Semantic segmentation
    print("\n[6/6] Processing panoramas...")

    if do_seg and data_dict.get('video_frame'):
        # Full pipeline: Semantic segmentation -> Point cloud -> Panorama
        print("Running semantic segmentation pipeline...")
        _run_segmentation_pipeline(extractor, config, viz)
    elif data_dict.get('pc_frame'):
        # Simplified pipeline: Generate panorama from point cloud (no segmentation)
        print("Generating panoramas from point clouds (no segmentation)...")
        extractor.convert_pc_to_global()
        extractor.generate_panoramas(window_sz=config.window_sz)

    # Get final data
    final_dict = extractor.get_data_dict()

    # Convert panorama format
    if 'pano_frame' in final_dict and len(final_dict['pano_frame']) > 0:
        pano = np.array(final_dict['pano_frame'])
        print(f"Panorama shape before conversion: {pano.shape}")

        # If 11 channels, convert to 5 channels
        if pano.ndim == 4 and pano.shape[-1] == 11:
            pano = convert_pano_to_5ch(pano)
            print(f"Panorama shape after 5ch conversion: {pano.shape}")

        # Crop height
        if pano.shape[2] > 140:
            pano = crop_pano_height(pano, target_height=140)
            print(f"Panorama shape after crop: {pano.shape}")

        final_dict['pano_frame'] = pano

    # Save
    print("\nSaving training set...")
    writer = TrainingSetWriter(output_dir)
    output_path = writer.save_minimal(
        final_dict,
        prefix='eDS20HZVZS',
        bag_name=bag_path
    )

    t_total_end = time.time()
    print(f"\n{'=' * 60}")
    print(f"Done! Total time: {t_total_end - t_total_start:.1f}s")
    print(f"Output: {output_path}")
    print("=" * 60)

    return output_path


def _run_segmentation_pipeline(
    extractor: DataExtractor,
    config: Config,
    viz: bool = False
):
    """
    Run semantic segmentation pipeline

    1. Perform DINOv2 segmentation on video frames
    2. Project segmentation results to point cloud
    3. Transform point cloud to global coordinate system
    4. Regenerate panorama (RGB + D + Seg)
    """
    from tqdm import tqdm
    from scipy.spatial.transform import Rotation as R

    data_dict = extractor.get_data_dict()

    # Load segmentation model
    print("Loading segmentation model...")
    try:
        labeler = SemanticLabeler(device='cuda')
        labeler.load_model()
    except Exception as e:
        print(f"Warning: Failed to load segmentation model: {e}")
        print("Falling back to panorama generation without segmentation")
        extractor.convert_pc_to_global()
        extractor.generate_panoramas(window_sz=config.window_sz)
        return

    # Find video frame indices that need segmentation (corresponding to point clouds)
    pc_t = np.array(data_dict['pc_t'])
    video_t = np.array(data_dict['video_t'])

    video_idxs = []
    for t in pc_t:
        idx = np.argmin(np.abs(video_t - t))
        video_idxs.append(idx)
    video_idxs = np.array(video_idxs)

    # Segment video frames
    print("Segmenting video frames...")
    seg_frames = []
    for idx in tqdm(video_idxs):
        frame = data_dict['video_frame'][idx][:, :, ::-1]  # RGB -> BGR for DINOv2
        seg = labeler.segment(frame)
        seg_frames.append(seg)

    # Project segmentation to point cloud, transform to global coordinate system
    print("Projecting segmentation to point clouds...")
    projector = SimpleLabelProjector(config.intrinsics)
    pc_processor = PointCloudProcessor(config)

    pc_frame_glob = []
    for i in tqdm(range(len(seg_frames))):
        video_idx = video_idxs[i]
        video_frame = data_dict['video_frame'][video_idx]
        depth_frame = data_dict['depth_frame'][i] if 'depth_frame' in data_dict else None

        if depth_frame is None:
            # Generate depth map from point cloud
            pc = data_dict['pc_frame'][i]
            depth_processor = DepthProcessor(config)
            depth_frame = depth_processor.pointcloud_to_depth(pc)[:, :, 0]

        # Ensure segmentation and depth are consistent
        seg = seg_frames[i] * np.sign(depth_frame[:, :, np.newaxis])

        # Project to point cloud
        pc_local = projector.project_seg_to_pointcloud(
            depth_frame, seg, video_frame, min_depth=0.3
        )

        # Transform to global coordinate system
        pos, quat = extractor.find_nearest_pose(pc_t[i])
        pc_world = pc_processor.camera_to_world(pc_local, pos, quat)

        pc_frame_glob.append([pc_world])  # Wrap in list to match original format

    # Generate panoramas
    print("Generating panoramas with segmentation...")
    pano_renderer = PanoramaRenderer(
        width=config.pano_width,
        height=config.pano_height,
        view_dist=config.view_dist
    )

    pano_frames = []
    pano_t = data_dict['pano_t']

    for i, t in enumerate(tqdm(pano_t)):
        pos, quat = extractor.find_nearest_pose(t)

        # Yaw matrix (horizon lock)
        euler = R.from_quat(quat).as_euler('ZYX')
        yaw_matrix = R.from_euler('ZYX', [euler[0], 0, 0]).as_matrix()

        # Window selection
        pc_idx = np.sum(pc_t < t)
        window_l = max(0, pc_idx - config.window_sz + 1)
        window_r = pc_idx + 1

        pc_window = pc_frame_glob[window_l:window_r]

        # Render panorama
        pano = pano_renderer.render_with_seg(
            pc_window, pos, yaw_matrix,
            filter_dist=True,
            sample_ratio=config.sample_ratio
        )
        pano_frames.append(pano)

        # Visualization
        if viz and i % 10 == 0:
            try:
                import matplotlib.pyplot as plt
                plt.clf()
                plt.subplot(121)
                plt.imshow(pano[:, :, :3].astype(np.uint8))
                plt.title('RGB')
                plt.subplot(122)
                plt.imshow(np.argmax(pano[:, :, 4:], axis=-1))
                plt.title('Segmentation')
                plt.pause(0.1)
            except:
                pass

    extractor.data.pano_frame = pano_frames


def main():
    parser = argparse.ArgumentParser(
        description='Data Pipeline V2 - Convert bag to training dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--bag', '-b', required=True,
        help='Path to ROS1 bag or ROS2 mcap file'
    )
    parser.add_argument(
        '--output', '-o', required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--no-seg', action='store_true',
        help='Skip semantic segmentation'
    )
    parser.add_argument(
        '--no-resample', action='store_true',
        help='Skip resampling'
    )
    parser.add_argument(
        '--hz', type=int, default=20,
        help='Resample frequency (default: 20)'
    )
    parser.add_argument(
        '--window-sz', type=int, default=32,
        help='Point cloud window size for panorama (default: 32)'
    )
    parser.add_argument(
        '--view-dist', type=float, default=10.0,
        help='View distance for panorama (default: 10.0)'
    )
    parser.add_argument(
        '--sample-ratio', type=float, default=0.3,
        help='Point cloud sample ratio (default: 0.3)'
    )
    parser.add_argument(
        '--viz', action='store_true',
        help='Show visualization'
    )
    parser.add_argument(
        '--save-video', action='store_true',
        help='Save video frames to output'
    )

    args = parser.parse_args()

    # Create configuration
    config = Config(
        resample_hz=args.hz,
        window_sz=args.window_sz,
        view_dist=args.view_dist,
        sample_ratio=args.sample_ratio
    )

    # Process
    process_bag(
        bag_path=args.bag,
        output_dir=args.output,
        config=config,
        do_seg=not args.no_seg,
        do_resample=not args.no_resample,
        viz=args.viz,
        save_video=args.save_video
    )


if __name__ == '__main__':
    main()
