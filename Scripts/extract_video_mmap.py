"""
Extract video frames from V20HZVZS and save as memory-mapped array.

Usage:
    python extract_video_mmap.py <dataset_name>

Example:
    python extract_video_mmap.py V2DataRedo_realsense0801_lag
"""

import os
import sys
import numpy as np
import joblib
from tqdm import tqdm
from PIL import Image

TRAINING_SETS_DIR = "/afs/cs.stanford.edu/u/weizhuo2/Documents/Data_pipe/Training_sets"
MMAP_DIR = "/afs/cs.stanford.edu/u/weizhuo2/Documents/fast_fs"

# Target size: divisible by 16 for DINO patches (15x27 patches)
TARGET_SIZE = (240, 432)  # H, W


def main(dataset_name):
    # Support both V20HZVZS and DS20HZVZS formats
    v20_path = f"{TRAINING_SETS_DIR}/V20HZVZS_{dataset_name}"
    prefix = "eDS20HZVZS"
    if not os.path.exists(v20_path):
        v20_path = f"{TRAINING_SETS_DIR}/DS20HZVZS_{dataset_name}"
        prefix = "DS20HZVZS"
    mmap_path = f"{MMAP_DIR}/{prefix}_{dataset_name}_video_uint8"

    print(f"Loading: {v20_path}")
    v20 = joblib.load(v20_path)
    video_frames = v20["video_frame"]
    n_frames = len(video_frames)
    orig_shape = video_frames[0].shape
    print(f"  {n_frames} frames, original shape: {orig_shape}")
    print(f"  Resizing to: {TARGET_SIZE}")

    print(f"Creating mmap: {mmap_path}")
    mmap_array = np.lib.format.open_memmap(
        mmap_path,
        mode="w+",
        dtype=np.uint8,
        shape=(n_frames, TARGET_SIZE[0], TARGET_SIZE[1], 3),
    )

    for i, frame in enumerate(tqdm(video_frames)):
        img = Image.fromarray(frame.astype(np.uint8))
        img_resized = img.resize((TARGET_SIZE[1], TARGET_SIZE[0]), Image.LANCZOS)
        mmap_array[i] = np.array(img_resized)

    mmap_array.flush()
    print("Done!")
    return mmap_array, prefix


def extract_dinov3(dataset_name, video=None, batch_size=32, prefix="eDS20HZVZS"):
    """Extract DINOv3 patch features, save as (N, 15, 27, 384)."""
    import torch
    from torchvision.transforms import v2

    out_path = f"{MMAP_DIR}/{prefix}_{dataset_name}_dinov3_patch_fp16"

    if video is None:
        video_path = f"{MMAP_DIR}/{prefix}_{dataset_name}_video_uint8"
        print(f"Loading video: {video_path}")
        video = np.load(video_path, mmap_mode="r")

    n_frames = len(video)
    h, w, c = video[0].shape
    print(f"  {n_frames} frames, shape: ({h}, {w}, {c})")

    print("Loading DINOv3 model...")
    model = torch.hub.load(
        "/afs/cs.stanford.edu/u/weizhuo2/Documents/gits/dinov3",
        "dinov3_vits16",
        source="local",
        pretrained=True,
        weights="/afs/cs.stanford.edu/u/weizhuo2/Documents/gits/dinov3/weights/dinov3_vits16_pretrain.pth",
    )
    model.cuda()
    model.eval()
    embed_dim = model.embed_dim

    # No resize needed - video is already 240x432
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    # Shape: (N, 15, 27, 384)
    print(f"Creating mmap: {out_path}")
    mmap_array = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float16, shape=(n_frames, 15, 27, embed_dim)
    )

    print(f"Extracting features (batch_size={batch_size})...")
    with torch.inference_mode():
        for i in tqdm(range(0, n_frames, batch_size)):
            batch_end = min(i + batch_size, n_frames)
            batch_tensor = torch.stack(
                [transform(f) for f in video[i:batch_end]]
            ).cuda()

            patch_features = model.get_intermediate_layers(
                batch_tensor, n=1, reshape=True, norm=True, return_class_token=False
            )[0]  # (B, 384, 15, 27)

            # (B, 384, 15, 27) -> (B, 15, 27, 384)
            mmap_array[i:batch_end] = (
                patch_features.permute(0, 2, 3, 1).cpu().numpy().astype(np.float16)
            )

    del mmap_array
    print("Done!")


if __name__ == "__main__":
    video, prefix = main(sys.argv[1])
    extract_dinov3(sys.argv[1], video=video, prefix=prefix)
