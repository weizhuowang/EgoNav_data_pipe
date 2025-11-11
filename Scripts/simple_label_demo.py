#!/usr/bin/env python3
"""
Minimal DINOv2 segmentation demo - labels a single image and displays it
"""
import sys
sys.path.append("/home/weizhuo2/Documents/gits/dinov2")

import numpy as np
import matplotlib.pyplot as plt
import urllib
from PIL import Image

import mmcv
from mmcv.runner import load_checkpoint
from mmseg.apis import init_segmentor, inference_segmentor
import dinov2.eval.segmentation_m2f.models.segmentors  # Register custom models
import time

# Segmentation channel mapping: ADE20k -> our 7 channels + background
ade2our = np.array([
    7, 3, 3, 7, 0, 4, 7, 0, 4, 4, 6, 4, 0, 5, 6, 2, 4, 7, 4, 4,
    4, 5, 4, 4, 4, 4, 3, 4, 3, 0, 6, 4, 4, 3, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 3, 4, 4, 4, 6, 4, 3, 4, 4, 7, 0, 1, 0, 4, 4, 4, 2,
    1, 4, 3, 4, 4, 4, 4, 4, 4, 6, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4,
    3, 5, 4, 7, 5, 3, 4, 7, 4, 3, 4, 5, 0, 4, 4, 0, 4, 1, 4, 4,
    4, 3, 0, 5, 5, 4, 4, 7, 4, 4, 4, 4, 4, 4, 3, 4, 4, 5, 4, 4,
    4, 4, 1, 4, 4, 4, 4, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 7, 4, 4, 4, 4, 7, 4, 4, 3, 4
])

# Channel definitions with colors for visualization
prompts = [
    ["ground, sidewalk", 0.2, 0.28, [0.1, 0.1, 0.1, 0.7]],  # dark
    ["stairs", 0.4, 0.4, [0.7, 0.7, 0.7, 0.7]],  # gray
    ["door, wood door, steel door, glass door, elevator door", 0.55, 0.5, [0.0, 0.7, 0.0, 0.7]],  # green
    ["wall, pillar", 0.47, 0.3, [0.7, 0.0, 0.0, 0.7]],  # red
    ["bin, chair, bench, desk, plants, curb, bushes, pole, tree", 0.55, 0.5, [0.0, 0.0, 0.7, 0.7]],  # blue
    ["people, person, pedestrian", 0.5, 0.5, [0.05, 0.7, 0.7, 0.7]],  # cyan
    ["grass, field, sand, hill, earth, dirt", 0.5, 0.5, [0.5, 0.5, 0.5, 0.7]],  # gray
]


def load_config_from_url(url: str) -> str:
    """Download config file from URL"""
    with urllib.request.urlopen(url) as f:
        return f.read().decode()


def show_mask(mask, image, color):
    """Overlay a segmentation mask on an image"""
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


def load_dinov2_model():
    """Load DINOv2 segmentation model"""
    print("Loading DINOv2 model...")

    DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
    CONFIG_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f_config.py"
    CHECKPOINT_URL = f"{DINOV2_BASE_URL}/dinov2_vitb14/dinov2_vitg14_ade20k_m2f.pth"

    cfg_str = load_config_from_url(CONFIG_URL)
    cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

    model = init_segmentor(cfg)
    load_checkpoint(model, CHECKPOINT_URL, map_location="cpu")
    model.cuda()
    model.eval()

    print("Model loaded!")
    return model


def segment_image(model, image_path):
    """Segment a single image and return visualization"""
    # Load image
    frame = mmcv.imread(image_path)  # BGR format
    print(f"Image shape: {frame.shape}")

    # Run inference
    print("Running segmentation...")
    for i in range(10):
        start_time = time.time()
        seg_ade20k = inference_segmentor(model, frame)[0]
        end_time = time.time()
        print(f"Segmentation time: {end_time - start_time} seconds")

    # Convert ADE20k labels to our channel definition
    seg_cls = np.zeros_like(seg_ade20k)
    for i in range(seg_ade20k.shape[0]):
        for j in range(seg_ade20k.shape[1]):
            seg_cls[i][j] = ade2our[seg_ade20k[i][j] + 1]

    # Create 7-channel segmentation mask
    seg_frame = np.zeros((frame.shape[0], frame.shape[1], 7))
    for i in range(len(prompts)):
        seg_frame[:, :, i] = seg_cls == i

    # Overlay masks on original image
    masked_source = frame.copy()
    for ch_idx in range(len(prompts)):
        color = prompts[ch_idx][3]
        masked_source = show_mask(
            seg_frame[:, :, ch_idx],
            masked_source,
            color=np.array([color[0], color[1], color[2], color[3] * 0.6]),
        )

    return masked_source, seg_frame


if __name__ == "__main__":
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Usage: python simple_label_demo.py <image_path>")
        print("Please provide an image path")
        sys.exit(1)

    # Load model
    model = load_dinov2_model()

    # Segment image
    result, seg_masks = segment_image(model, image_path)

    # Display results
    plt.figure(figsize=(12, 6))

    # Original image
    plt.subplot(1, 2, 1)
    original = mmcv.imread(image_path)
    plt.imshow(original[:, :, ::-1])  # BGR to RGB
    plt.title("Original Image")
    plt.axis("off")

    # Segmented image
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.title("Segmentation Result")
    plt.axis("off")

    # Add legend
    legend_labels = [p[0] for p in prompts]
    legend_colors = [p[3] for p in prompts]
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=tuple(c[:3]) + (0.6,), label=l) for l, c in zip(legend_labels, legend_colors)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.show()

    print("Done!")
