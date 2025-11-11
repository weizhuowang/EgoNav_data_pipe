#!/usr/bin/env python3
"""
Profile DINOv2 backbone and Mask2Former head separately
"""
import sys
sys.path.append("/home/weizhuo2/Documents/gits/dinov2")

import numpy as np
import time
import torch
import urllib

import mmcv
from mmcv.runner import load_checkpoint
from mmseg.apis import init_segmentor
import dinov2.eval.segmentation_m2f.models.segmentors


def load_config_from_url(url: str) -> str:
    """Download config file from URL"""
    with urllib.request.urlopen(url) as f:
        return f.read().decode()


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


def profile_model(model, image_path, num_runs=10):
    """Profile the model components"""
    # Load image
    frame = mmcv.imread(image_path)
    print(f"Image shape: {frame.shape}")

    # Prepare input using the model's internal methods
    from mmseg.apis.inference import LoadImage
    from mmseg.datasets.pipelines import Compose
    from mmcv.parallel import collate, scatter

    cfg = model.cfg
    device = next(model.parameters()).device
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    data = dict(img=image_path)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    data = scatter(data, [device])[0]

    # Warmup
    print("\nWarming up...")
    with torch.no_grad():
        for _ in range(3):
            _ = model.encode_decode(data['img'][0], data['img_metas'][0])

    torch.cuda.synchronize()

    # Profile full pipeline
    print(f"\nProfiling full pipeline ({num_runs} runs)...")
    full_times = []
    with torch.no_grad():
        for i in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()
            _ = model.encode_decode(data['img'][0], data['img_metas'][0])
            torch.cuda.synchronize()
            end = time.time()
            full_times.append(end - start)
            print(f"Run {i+1}: {(end-start)*1000:.2f} ms")

    print(f"\nFull pipeline average: {np.mean(full_times)*1000:.2f} ms ± {np.std(full_times)*1000:.2f} ms")

    # Profile backbone only
    print(f"\nProfiling DINOv2 backbone ({num_runs} runs)...")
    backbone_times = []
    with torch.no_grad():
        for i in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()
            x = model.backbone(data['img'][0])
            torch.cuda.synchronize()
            end = time.time()
            backbone_times.append(end - start)
            print(f"Run {i+1}: {(end-start)*1000:.2f} ms")

    print(f"\nBackbone average: {np.mean(backbone_times)*1000:.2f} ms ± {np.std(backbone_times)*1000:.2f} ms")

    # Profile decode head only (using cached backbone features)
    print(f"\nProfiling M2F decode head ({num_runs} runs)...")
    with torch.no_grad():
        x = model.backbone(data['img'][0])
        head_times = []
        for i in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()
            _ = model.decode_head.forward_test(x, data['img_metas'][0], model.test_cfg)
            torch.cuda.synchronize()
            end = time.time()
            head_times.append(end - start)
            print(f"Run {i+1}: {(end-start)*1000:.2f} ms")

    print(f"\nDecode head average: {np.mean(head_times)*1000:.2f} ms ± {np.std(head_times)*1000:.2f} ms")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Full pipeline:     {np.mean(full_times)*1000:.2f} ms")
    print(f"DINOv2 backbone:   {np.mean(backbone_times)*1000:.2f} ms ({np.mean(backbone_times)/np.mean(full_times)*100:.1f}%)")
    print(f"M2F decode head:   {np.mean(head_times)*1000:.2f} ms ({np.mean(head_times)/np.mean(full_times)*100:.1f}%)")
    print(f"Overhead:          {(np.mean(full_times)-np.mean(backbone_times)-np.mean(head_times))*1000:.2f} ms")
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Usage: python profile_model.py <image_path>")
        sys.exit(1)

    # Load model
    model = load_dinov2_model()

    # Profile
    profile_model(model, image_path, num_runs=10)
