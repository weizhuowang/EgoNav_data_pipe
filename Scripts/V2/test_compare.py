#!/usr/bin/env python3
"""
测试脚本：加载 V2 输出和原始 eDS 进行对比
可以在这里打断点查看数据内容
"""

import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

# 路径
# V2_PATH = "/arm/u/weizhuo2/Documents/Data_pipe/Training_sets/V2_test/V2TEST_PANO500_V2DataRedo_field_lag"
V2_PATH = "/arm/u/weizhuo2/Documents/Data_pipe/Training_sets/V2_test/V2TEST7_V2DataRedo_field_lag"
EDS_PATH = (
    "/arm/u/weizhuo2/Documents/Data_pipe/Training_sets/eDS20HZVZS_V2DataRedo_field_lag"
)


def load_data():
    """加载两个数据集"""
    print("Loading V2 data...")
    v2 = joblib.load(V2_PATH)

    print("Loading eDS data...")
    eds = joblib.load(EDS_PATH)

    print("V2 keys:", sorted(v2.keys()))
    print("eDS keys:", sorted(eds.keys()))

    return v2, eds


def print_keys(data, name):
    """打印数据集的 keys 和形状"""
    print(f"\n=== {name} ===")
    for k, v in sorted(data.items()):
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, list):
            if len(v) > 0 and isinstance(v[0], np.ndarray):
                print(f"  {k}: list[{len(v)}], elem shape={v[0].shape}")
            else:
                print(f"  {k}: list[{len(v)}]")
        else:
            print(f"  {k}: {type(v).__name__}")


def print_pano_stats(pano, name):
    """打印 pano_frame 各通道统计"""
    print(f"\n=== {name} pano_frame 通道统计 ===")
    print(f"shape: {pano.shape}")
    for i in range(pano.shape[-1]):
        ch = pano[..., i]
        print(f"  ch{i}: min={ch.min():.3f}, max={ch.max():.3f}, mean={ch.mean():.3f}")


def compare_data_array(v2, eds):
    """对比 data_array"""
    da_v2 = v2["data_array"]
    da_eds = eds["data_array"]

    print(f"\n=== data_array 对比 ===")
    print(f"V2:  shape={da_v2.shape}")
    print(f"eDS: shape={da_eds.shape}")

    # 对比前 N 行
    n = min(len(da_v2), len(da_eds))
    diff = np.abs(da_v2[:n] - da_eds[:n])
    print(f"前 {n} 行最大差异: {diff.max():.6f}")
    print(f"前 {n} 行平均差异: {diff.mean():.6f}")


def main():
    # 加载数据 - 在这里打断点可以查看 v2 和 eds
    v2, eds = load_data()

    # 打印 keys
    print_keys(v2, "V2")
    print_keys(eds, "eDS")

    # pano_frame 统计
    print_pano_stats(v2["pano_frame"], "V2")
    print_pano_stats(eds["pano_frame"], "eDS")

    # data_array 对比
    compare_data_array(v2, eds)

    # === 在这里打断点，可以交互式查看数据 ===
    # 例如:
    #   v2['pano_frame'][0]  - V2 第一帧全景
    #   eds['pano_frame'][0] - eDS 第一帧全景
    #   v2['data_array'][:10] - V2 前10行 data_array

    # 绘制对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # pano_frame 对比 (取第一帧的 RGB 通道)
    idx = 0
    pano_v2 = v2["pano_frame"][idx, :, :, 0:3].astype(np.uint8)  # RGB
    pano_eds = eds["pano_frame"][idx, :, :, 0:3].astype(np.uint8)

    axes[0, 0].imshow(pano_v2)
    axes[0, 0].set_title(f"V2 pano_frame[{idx}] RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pano_eds)
    axes[0, 1].set_title(f"eDS pano_frame[{idx}] RGB")
    axes[0, 1].axis("off")

    # depth_frame 对比 (取第一帧)
    depth_v2 = v2["depth_frame"][idx].squeeze()
    depth_eds = eds["depth_frame"][idx].squeeze()

    axes[1, 0].imshow(depth_v2, cmap="viridis")
    axes[1, 0].set_title(f"V2 depth_frame[{idx}]")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(depth_eds, cmap="viridis")
    axes[1, 1].set_title(f"eDS depth_frame[{idx}]")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()

    print("Done")


if __name__ == "__main__":
    main()
