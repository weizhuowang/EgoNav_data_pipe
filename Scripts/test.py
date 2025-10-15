import numpy as np
import pickle, time, os
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import bz2, gzip, joblib

os.environ["PATH"] = "/usr/bin:" + os.environ["PATH"]
import skvideo.io

# skvideo.io.setFFmpegPath("/usr/bin/ffmpeg")
from scipy.spatial.transform import Rotation as R
import cv2

prompts = [
    ["ground, sidewalk", 0.2, 0.28, [0.1, 0.1, 0.1, 0.7]],  # dark
    ["stairs", 0.4, 0.4, [0.7, 0.7, 0.7, 0.7]],  # gray ish
    #    ["ramp",0.3,0.25,[0.5, 0.5, 0.5, 0.7]],   # dark ish
    [
        "door, wood door, steel door, glass door, elevator door",
        0.55,
        0.5,
        [0.0, 0.7, 0.0, 0.7],
    ],  # green
    ["wall, pillar", 0.47, 0.3, [0.7, 0.0, 0.0, 0.7]],  # red
    [
        "bin, chair, bench, desk, plants, curb, bushes, pole, tree",
        0.55,
        0.5,
        [0.0, 0.0, 0.7, 0.7],
    ],
    ["people, person, pedestrian", 0.5, 0.5, [0.05, 0.7, 0.7, 0.7]],
    ["Rough Ground", 0.5, 0.5, [0.5, 0.5, 0.5, 0.7]],
]


def pano2pc(
    raw_pano, curr_pos, curr_quat, semantic_class=None, color=False, rem_isl=True
):
    if np.max(raw_pano) > 1:
        raise ValueError("panorama depth should be in range 0,1")

    if rem_isl:
        raw_pano = remove_islands(
            raw_pano, area_thres=70, semantic_class=semantic_class
        )

    # Preprocess panorama
    pano_height, pano_width = raw_pano.shape[0], raw_pano.shape[1]
    pano = np.zeros((pano_width // 2, pano_width)) + 1.0
    pano[(pano.shape[0] - pano_height) :, :] = raw_pano
    pano = (pano / np.max(pano) * 255).astype(np.uint8)

    # Precompute rotations
    r = R.from_quat(curr_quat).as_euler("ZYX")  # Z rotation only
    pov_rot = R.from_euler("ZYX", [r[0], 0, 0]).as_matrix()

    scale_fac = 360 / pano_width
    i_indices, j_indices = np.indices(pano.shape)
    valid_indices = pano < 254

    # Calculate depth
    d = pano[valid_indices] / 255.0 * 10

    # Calculate rotations for each pixel
    pixel_rotations = R.from_euler(
        "ZYX",
        np.vstack(
            [
                180 - j_indices[valid_indices] * scale_fac,
                i_indices[valid_indices] * scale_fac - 90,
                np.zeros_like(d),
            ]
        ).T,
        degrees=True,
    ).apply(np.array([1, 0, 0]))

    # Compute point cloud
    pc = pov_rot @ pixel_rotations.T * d + curr_pos.reshape(3, 1)

    # Handle color if needed
    if color:
        pc_idx = np.ravel_multi_index(
            (i_indices[valid_indices], j_indices[valid_indices]), pano.shape
        )
        return pc.T, pc_idx
    else:
        return pc.T


"""
Remove artifacts in the depth image by looking at the individual islands.
input: depth_image np array [h,w]
output: np array [h,w] in range 0-255
"""


def remove_islands(depth_image_given, semantic_class=None, area_thres=70):
    depth_image_raw = depth_image_given.copy()
    if np.max(depth_image_raw) < 1.1:
        depth_image_raw = np.round(depth_image_raw * 255)

    # If semantic class is provided, only consider island in wall and obs
    if semantic_class is not None:
        notrelevant_msk = np.isin(semantic_class, [0, 1, 2, 5, 6])
        depth_image_notrelevant = depth_image_raw[notrelevant_msk].copy()
        depth_image_raw[notrelevant_msk] = 255

    depth_image_msk = depth_image_raw.copy()
    depth_image_msk = depth_image_msk.astype("uint8")

    # Set the known region (0-255) and unknown region (255)
    depth_image_msk[depth_image_msk == 255] = 0  # Set unknown regions to 0 temporarily
    depth_image_msk[depth_image_msk > 0] = 1  # Set known regions to 1

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        depth_image_msk, 4, cv2.CV_32S
    )

    # Iterate over connected components and remove small ones
    for i in range(1, num_labels):  # Start from 1 to ignore the background
        size = stats[i, cv2.CC_STAT_AREA]
        if size < area_thres:  # Threshold for island size
            depth_image_msk[labels == i] = 0

    depth_image_raw[depth_image_msk == 0] = 255

    # put back the not relevant region
    if semantic_class is not None:
        depth_image_raw[notrelevant_msk] = depth_image_notrelevant
        depth_image_raw = remove_islands(depth_image_raw, area_thres=area_thres)

    if np.max(depth_image_given) < 1.1:
        depth_image_raw = depth_image_raw / 255.0
    return depth_image_raw


def check_saved_set():
    import joblib, cv2

    data_dict = joblib.load(
        "/sailhome/weizhuo2/Documents/Data_pipe/Bags/20HZVZS_V2DataRedo_realsense0801_lag"
    )
    pano_frames = np.array(data_dict["pano_frame"])
    cv2.namedWindow("Grayscale Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Grayscale Image", 720, 360)  # Set the initial window size

    for i in tqdm(range(pano_frames.shape[0])):
        # resized_image = cv2.resize(pano_frames[i], (720, 360))
        cv2.imshow("Grayscale Image", pano_frames[i])
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    print(data_dict.keys())


# DS set
def check_saved_Dset():
    import joblib, cv2

    data_dict = joblib.load(
        "/sailhome/weizhuo2/Documents/Data_pipe/Training_sets/DS20HZVZS_V2DataRedo_field_lag"
    )
    pano_frames = data_dict["pano_frame"]
    cv2.namedWindow("Grayscale Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Grayscale Image", 720, 360)  # Set the initial window size

    for i in tqdm(range(len(pano_frames))):
        cv2.imshow("Grayscale Image", data_dict["pano_frame"][i])
        cv2.waitKey(1)


def alpha_blend(original_image, mask, color, alpha_value):
    color_image = np.ones(original_image.shape, dtype=original_image.dtype) * np.array(
        color
    )
    alpha_channel = mask * alpha_value
    blended_image = (
        alpha_channel[..., None] * color_image
        + (1 - alpha_channel[..., None]) * original_image
    )
    return blended_image.astype(original_image.dtype)


# Test wich write method takes the least ram
def write_to_disk():
    # Load test data
    t1 = time.time()
    fpath = "/afs/cs.stanford.edu/u/weizhuo2/Documents/Data_pipe/Training_sets/DS20HZVZS_field_2021-12-09-16-45-58lag"
    infile = open(fpath, "rb")
    data_dict = pickle.load(infile, encoding="latin1")
    infile.close()
    t2 = time.time()
    print("Loading took {:.2f}s".format(t2 - t1))

    # Save to disk
    t3 = time.time()
    save_path = (
        "/afs/cs.stanford.edu/u/weizhuo2/Documents/Data_pipe/Training_sets/test_write"
    )
    # outfile = open(save_path,'wb')           # Fastest 57s 4300MB
    # outfile = bz2.BZ2File(save_path, 'w')    # Least space 473s 341MB
    # outfile = gzip.open(save_path,'wb')      # most economical 310s 493MB
    # pickle.dump(data_dict,outfile)
    # outfile.close()

    joblib.dump(
        data_dict, save_path, compress=("lz4", 1)
    )  # THE BEST!!! const mem, 500M, 130s
    t4 = time.time()
    print("Saving took {:.2f}s".format(t4 - t3))


def visualize_mmap():
    from scipy.spatial import KDTree
    import open3d as o3d

    def int2onehot(seg_frame):
        seg_frame_full_range = seg_frame
        seg_frame_onehot = np.zeros((seg_frame.shape[0], seg_frame.shape[1], 7))
        for i in range(7):
            seg_frame_onehot[:, :, [i]] = seg_frame_full_range == i
        return seg_frame_onehot

    def composite_segrgb(raw_pano_frame):
        pano_frame = raw_pano_frame.copy()
        rgb_frame = pano_frame[:, :, :3] / 255.0
        seg_frame = int2onehot(pano_frame[:, :, [4]])
        composited_frame = rgb_frame.copy()
        for j in range(seg_frame.shape[2]):
            color = prompts[j][3][:3]
            composited_frame = alpha_blend(
                composited_frame, seg_frame[:, :, j], color, 0.55
            )
        return composited_frame

    loaded_prefix = "/sailhome/weizhuo2/Documents/transfer/"
    len_dict = joblib.load(loaded_prefix + "len_dict.lz4")

    raw_data_path = (
        "../Data_pipe/Training_sets/" + "DS20HZVZS_realsense_2022-08-01-17-25-56_lag"
    )
    mmap_path = loaded_prefix + raw_data_path.split("/")[-1]
    # len_data,len_pano = len_dict[raw_data_path]
    # data_array = np.memmap(mmap_path+'_da', dtype='float32', mode='r', shape=(len_data,25))
    # pano = np.memmap(mmap_path+'_pano_uint8', dtype='uint8', mode='r', shape=(len_pano,5,140,360))

    len_data = 6790
    data_array = np.memmap(
        mmap_path + "_da", dtype="float32", mode="r", shape=(len_data, 25)
    )
    pano = np.memmap(
        mmap_path + "_pano_uint8", dtype="uint8", mode="r", shape=(5876, 5, 140, 360)
    )

    cv2.namedWindow("Seg Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Seg Image", 1650, 660)  # Set the initial window size

    for i in tqdm(range(0, len_data, 7)):
        pano_idx = int(data_array[i, -2])
        pano_frame = pano[pano_idx].copy()

        depth_clean = remove_islands(
            pano_frame[3], semantic_class=pano_frame[4], area_thres=70
        )

        # pano_frame[:3,depth_clean==255] = 0
        # pano_frame[3,depth_clean==255] = 255
        # pano_frame[4,depth_clean==255] = 7

        img = composite_segrgb(pano_frame.transpose(1, 2, 0))
        # img = pano_frame[:3].transpose(1,2,0)/255.0
        cv2.imshow("Seg Image", img[:, :, [2, 1, 0]])
        cv2.waitKey(1)

        # scene_pc = pano2pc(pano_frame[3]/255.0, np.array([0,0,0]), np.array([0,0,0,1]), semantic_class=pano_frame[4], color=False, rem_isl=True)


# visualize the 3d reprojected segmented panorama
def visualize(save_mp4=False):
    import cv2

    if save_mp4:
        fps = 20
        video_name = (
            "/afs/cs.stanford.edu/u/weizhuo2/Documents/Data_pipe/Training_sets/"
            + "dinov2_example.mp4"
        )
        out = skvideo.io.FFmpegWriter(
            video_name,
            inputdict={"-r": str(fps)},
            outputdict={
                "-r": str(fps),
                "-vcodec": "libx264",  # use the h.264 codec
                "-crf": "20",  # set the constant rate factor to 20, which is visually lossless
                "-preset": "veryslow",  # the slower the better compression, in princple, try
                #                         #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
            },
            verbosity=1,
        )

    prompts = [
        ["ground, sidewalk", 0.2, 0.28, [0.1, 0.1, 0.1, 0.7]],  # dark
        ["stairs", 0.4, 0.4, [0.7, 0.7, 0.7, 0.7]],  # gray ish
        #    ["ramp",0.3,0.25,[0.5, 0.5, 0.5, 0.7]],   # dark ish
        [
            "door, wood door, steel door, glass door, elevator door",
            0.55,
            0.5,
            [0.0, 0.7, 0.0, 0.7],
        ],  # green
        ["wall, pillar", 0.47, 0.3, [0.7, 0.0, 0.0, 0.7]],  # red
        [
            "bin, chair, bench, desk, plants, curb, bushes, pole, tree",
            0.55,
            0.5,
            [0.0, 0.0, 0.7, 0.7],
        ],
        ["people, person, pedestrian", 0.5, 0.5, [0.05, 0.7, 0.7, 0.7]],
        ["Rough Ground", 0.5, 0.5, [0.5, 0.5, 0.5, 0.7]],
    ]

    fpath = "/afs/cs.stanford.edu/u/weizhuo2/Documents/Data_pipe/Training_sets/DS20HZVZS_field_2021-12-09-16-45-58lag"
    # fpath = '/sailhome/weizhuo2/Documents/Data_pipe/Training_sets/DS20HZVZS_human_2022-04-08-14-23-31_lag'
    fpath = "/sailhome/weizhuo2/Documents/Data_pipe/Training_sets/DS20HZVZS_V2DataRedo_field_lag"
    fpath = "/sailhome/weizhuo2/Documents/Data_pipe/Training_sets/eDS20HZVZS_V2DataRedo_realsense0801_lag"
    # fpath = '/sailhome/weizhuo2/Documents/Data_pipe/Training_sets/eDS20HZVZS_V2DataNew_240105cactusC_lag'
    # fpath = '/sailhome/weizhuo2/Documents/Data_pipe/Training_sets/eDS20HZVZS_V2DataNew_240103flomoC_lag'
    # fpath = '/sailhome/weizhuo2/Documents/Data_pipe/Training_sets/DS20HZVZS_V2DataNew_231228quad_lag'
    # fpath = '/sailhome/weizhuo2/Documents/Data_pipe/Training_sets/DS20HZVZS_V3DataRedo_2024-01-27-14-30-20_lag'
    fpath = "/sailhome/weizhuo2/Documents/Data_pipe/Training_sets/eDS20HZVZS_leo"
    # infile = open(fpath,'rb')
    # data_dict = pickle.load(infile, encoding='latin1')
    # infile.close()
    t1 = time.time()
    data_dict = joblib.load(fpath)
    print(data_dict["prompts"])
    # prompts = data_dict['prompts']
    t2 = time.time()
    print("data_loaded ", t2 - t1, "s")
    pano_idx = data_dict["data_array"][:, -2].astype(int)[::3]

    # cv2 window faster
    cv2.namedWindow("Seg Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Seg Image", 1440, 720)  # Set the initial window size

    # for i in tqdm(range(0,len(data_dict['pano_frame'])-100)):
    tlst = time.time()
    for i in tqdm(pano_idx):
        rgb_frame = data_dict["pano_frame"][i][:, :, :3] / 255
        seg_frame = data_dict["pano_frame"][i][:, :, 4:]
        composited_frame = rgb_frame.copy()
        for j in range(seg_frame.shape[2]):
            color = prompts[j][3][:3]
            composited_frame = alpha_blend(
                composited_frame, seg_frame[:, :, j], color, 0.4
            )

        if save_mp4:
            data = np.uint8(composited_frame * 255)
            out.writeFrame(data)
        else:
            # plt.clf()
            # plt.imshow(composited_frame)
            # # plt.imshow()
            # plt.pause(0.05)
            cv2.imshow("Seg Image", rgb_frame[:, :, [2, 1, 0]])
            t_pause = int(max(1, 50 - (time.time() - tlst) * 1000)) + 100
            cv2.waitKey(t_pause)
            tlst = time.time()

    if save_mp4:
        out.close()
    else:
        # plt.show()
        pass


# Save video frame for dinov2 label app
def pick_video_frames():
    # load all training set and randomly pick 500 images out of all the video_frames
    fpath = "/afs/cs.stanford.edu/u/weizhuo2/Documents/Data_pipe/Training_sets"
    all_files = [
        "V20HZVZS_realsense_2022-08-01-17-25-56_lag",
        "V20HZVZS_realsense_2022-07-12-19-26-18_lag",
        "V20HZVZS_human_2022-04-08-14-23-31_lag",
        "V20HZVZS_human_2022-04-01-16-30-44_lag",
        "V20HZVZS_human_2022-03-06-14-51-06_lag",
        "V20HZVZS_field_2021-12-09-16-45-58lag",
    ]
    video_frames = []
    for file in tqdm(all_files):
        infile = open(os.path.join(fpath, file), "rb")
        data_dict = pickle.load(infile, encoding="latin1")
        infile.close()
        print(data_dict.keys())
        video_frames.extend(data_dict["video_frame"][150:-600])
        print(len(video_frames))
        del data_dict

    print("Saving")
    selected_idx = np.random.permutation(np.arange(len(video_frames)))[:500]
    selected_frames = [video_frames[i] for i in selected_idx]
    outfile = open(os.path.join(fpath, "selected_video_frames"), "wb")
    pickle.dump(selected_frames, outfile)
    outfile.close()


def visualize_Gdino():
    prompts = [
        ["ground, sidewalk", 0.2, 0.28, [0.1, 0.1, 0.1, 0.7]],  # dark
        ["stairs", 0.4, 0.4, [0.7, 0.7, 0.7, 0.7]],  # gray ish
        #    ["ramp",0.3,0.25,[0.5, 0.5, 0.5, 0.7]],   # dark ish
        [
            "door, wood door, steel door, glass door, elevator door",
            0.55,
            0.5,
            [0.0, 0.7, 0.0, 0.7],
        ],  # green
        ["wall, pillar", 0.47, 0.3, [0.7, 0.0, 0.0, 0.7]],  # red
        [
            "bin, chair, bench, desk, plants, curb, bushes, pole, tree",
            0.55,
            0.5,
            [0.0, 0.0, 0.7, 0.7],
        ],
        ["people, person, pedestrian", 0.5, 0.5, [0.0, 0.0, 0.7, 0.7]],
        ["Rough Ground", 0.5, 0.5, [0.5, 0.5, 0.5, 0.7]],
    ]

    fpath = "/arm/u/weizhuo2/Documents/Data_pipe/Training_sets/S20HZVZS_realsense_2022-08-01-17-25-56_lag"
    data_dict = joblib.load(fpath)
    pano = data_dict["pano_frame"]
    for i in range(300, 1000, 1):
        print(i)
        rgb_frame = data_dict["pano_frame"][i][:, :, :3] / 255
        seg_frame = np.sign(data_dict["pano_frame"][i][:, :, 4:])
        composited_frame = rgb_frame.copy()
        for j in range(seg_frame.shape[2]):
            color = prompts[j][3][:3]
            composited_frame = alpha_blend(
                composited_frame, seg_frame[:, :, j], color, 0.4
            )
        plt.imshow(composited_frame)
        plt.pause(0.05)


def remove_edges():
    data_fname = "/sailhome/weizhuo2/Documents/Data_pipe/Training_sets/V20HZVZS_V2DataRedo_realsense0801_lag"
    data_dict = joblib.load(data_fname)
    depth_frame = np.array(data_dict["depth_frame"]).repeat(3, axis=3)
    for i in range(len(depth_frame)):
        a = depth_frame[i]
        # # Gradient based
        # sobelx = cv2.Sobel(a, cv2.CV_64F, 1, 0, ksize=5)
        # sobely = cv2.Sobel(a, cv2.CV_64F, 0, 1, ksize=5)

        # edges = np.hypot(sobelx, sobely*0)
        # edges = np.uint8(edges / edges.max() * 255)

        # Canny edge detection
        edges = cv2.Canny(np.round(a * 255).astype(np.uint8), 25, 32)
        cv2.imshow("Depth Frame", a)
        cv2.imshow(
            "Depth Frame edges",
            a
            + 1.0
            / 255
            * np.pad(
                edges[:, :, np.newaxis],
                ((0, 0), (0, 0), (2, 0)),
                "constant",
                constant_values=0,
            ),
        )

        # remove edges
        kernel = np.ones((10, 10))
        eroded_edges = cv2.erode(edges - 1, kernel) + 1
        eroded_edges = eroded_edges[:, :, np.newaxis].repeat(3, axis=2)
        cv2.imshow("Depth Frame eroded edges", eroded_edges)

        processed_frame = a.copy()
        processed_frame[eroded_edges > 0] = 0

        cv2.imshow("Depth Frame processed", processed_frame)
        cv2.waitKey(1)


def read_jk_data():
    fpath = "/afs/cs.stanford.edu/u/weizhuo2/Documents/Data_pipe/Training_sets/Backup/train.pkl"
    infile = open(fpath, "rb")
    data_dict = pickle.load(infile, encoding="latin1")
    infile.close()
    print(len(data_dict.keys()))

    print(
        data_dict[0].keys()
    )  # ['camera_wearer_traj', 'nearby_people', 'semantics_encoding', 'depth_encoding']


def check_DINO_dataset():
    loaded_prefix = "/arm/u/weizhuo2/Documents/Data_pipe/Training_sets/"

    raw_data_path = "eDS20HZVZS_V2DataRedo_realsense0801_lag"
    data_path = loaded_prefix + raw_data_path

    data_dict = joblib.load(data_path)
    print(data_dict.keys())


def check_addvideo_dataset():
    loaded_prefix = "/arm/u/weizhuo2/Documents/Data_pipe/Training_sets/"

    raw_data_path = "V20HZVZS_V2DataRedo_realsense0801_lag"
    raw_data_path = "V20HZVZS_leo"
    data_path = loaded_prefix + raw_data_path

    data_dict = joblib.load(data_path)
    print(data_dict.keys())


def check_readbag_dataset():
    loaded_prefix = "/arm/u/weizhuo2/Documents/Data_pipe/Training_sets/"

    raw_data_path = "20HZVZS_V2DataRedo_realsense0801_lag"
    data_path = loaded_prefix + raw_data_path

    data_dict = joblib.load(data_path)
    print(data_dict.keys())


def check_mmap_dataset():
    loaded_prefix = "/arm/u/weizhuo2/Documents/Data_pipe/Training_sets/mmaps/"
    len_dict = joblib.load(loaded_prefix + "len_dict.lz4")

    raw_data_path = "DS20HZVZS_V3DataRedo_realsense0801C_lag"
    mmap_path = loaded_prefix + raw_data_path

    len_data = len_dict["../Data_pipe/Training_sets/" + raw_data_path]
    data_array = np.memmap(
        mmap_path + "_da", dtype="float32", mode="r", shape=(len_data[0], 25)
    )
    pano = np.memmap(
        mmap_path + "_pano_uint8",
        dtype="uint8",
        mode="r",
        shape=(len_data[1], 5, 140, 360),
    )

    print(data_array.shape, pano.shape)


if __name__ == "__main__":
    # visualize_mmap()

    visualize()
    # pick_video_frames()
    # write_to_disk()
    # visualize_Gdino()
    # check_saved_set()
    # check_saved_Dset()
    # remove_edges()

    # read_jk_data()

    # ========check datasets========

    # check_readbag_dataset()
    # check_addvideo_dataset()
    # check_DINO_dataset()
    # check_mmap_dataset()
