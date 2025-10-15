import os

os.environ["PATH"] = "/usr/bin:" + os.environ["PATH"]

import rosbag
import sys
import yaml
import numpy as np
import ros_numpy as rnp
import matplotlib.pyplot as plt
import sensor_msgs

# import cv2
import skvideo.io
from tqdm import tqdm

"""
This script is used to extract video frames from a rosbag file. Not a part of 
normal data pipeline.
"""

# ========================
# Helper functions
# ========================


# ========================
# Helper classes
# ========================


class video_generator:
    def __init__(self, fpath):
        self.fpath = fpath
        self.bag = rosbag.Bag(self.fpath)
        self.info_dict = yaml.safe_load(self.bag._get_yaml_info())

        self.video_t = []
        self.video_frame = []

    # Clean up function
    def exit(self):
        self.bag.close()

    def collect_video(self, video_channel):
        # for each message, populate the training set array
        self.video_channel = video_channel
        for topic, msg, t in tqdm(self.bag.read_messages()):
            self.extract_data(topic, msg, t)

    def extract_data(self, topic, msg, t):
        new_entry = False
        if topic == self.video_channel:
            print("[TYPE] video_frame", self.video_channel, t, len(self.video_t))
            # Convert to cv frame
            msg.__class__ = sensor_msgs.msg._Image.Image
            img = rnp.numpify(msg)

            # Gather info from header
            msg_t = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9

            # Save to respective array
            self.video_t.append(msg_t)
            self.video_frame.append(img)

            # print('[DEBUG] ',msg.height,msg.width)

            # Draw frame
            # plt.clf()
            # plt.imshow(img)
            # plt.pause(0.000001)

        else:
            # print('[ALERT] Ununsed',topic,type(msg),t,)
            pass

    def save_video(self, save_fpath):
        size = self.video_frame[0].shape[0], self.video_frame[0].shape[1]
        # nframes = len(self.video_t)
        # fps     = round(nframes/((self.video_t[-1]-self.video_t[0])),1)
        fps = 30
        print(self.video_t[-1] - self.video_t[0])
        nframes = int((self.video_t[-1] - self.video_t[0]) * fps)
        self.video_t_n = np.array(self.video_t) - self.video_t[0]

        if self.video_channel == "/d400/color/image_raw":
            video_name = save_fpath + "_video.mp4"
            colored = True
            out = skvideo.io.FFmpegWriter(
                video_name,
                inputdict={"-r": str(fps)},
                outputdict={
                    "-r": str(fps),
                    "-vcodec": "libx264",  # use the h.264 codec
                    "-crf": "20",  # set the constant rate factor to 20, which is visually lossless
                    "-pix_fmt": "yuv420p",  # use the yuv420p pixel format for most video players
                    # '-preset':'veryslow'   #the slower the better compression, in princple, try
                    #                         #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
                },
                verbosity=1,
            )
        elif self.video_channel == "/testpano":
            video_name = save_fpath + "_pano.mp4"
            colored = False
            out = skvideo.io.FFmpegWriter(
                video_name,
                inputdict={"-r": str(fps)},
                outputdict={
                    "-r": str(
                        fps
                    ),  # doesn't actually affect anything... only in meta data
                    "-vcodec": "libx264",  # use the h.264 codec
                    "-crf": "0",  # set the constant rate factor to 0, which is lossless
                    "-pix_fmt": "yuv420p",  # use the yuv420p pixel format for most video players
                    # '-preset':'veryslow'   #the slower the better compression, in princple, try
                    #                         #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
                },
            )
        else:
            video_name = save_fpath + "_other.mp4"
            colored = True
            out = skvideo.io.FFmpegWriter(
                video_name,
                inputdict={"-r": str(fps)},
                outputdict={
                    "-r": str(fps),
                    "-vcodec": "libx264",  # use the h.264 codec
                    "-crf": "25",  # set the constant rate factor to 18, which is visually lossless
                    "-pix_fmt": "yuv420p",  # use the yuv420p pixel format for most video players
                    # '-preset':'veryslow'   #the slower the better compression, in princple, try
                    #                         #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
                },
                verbosity=1,
            )

        # out    = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]),colored)

        for i in tqdm(range(1, nframes)):
            # print(i,'/',nframes)
            # video_idx = sum((self.video_t_n-(1/fps)*i)<=0)
            video_idx = np.searchsorted(self.video_t_n, (1 / fps) * i)
            data = self.video_frame[video_idx]
            if colored:
                out.writeFrame(data)  # cv want rgb, ros has bgr
            else:
                color_data = np.repeat(data[:, :, np.newaxis], 3, axis=2)
                out.writeFrame(color_data)
        # out.release()
        out.close()


# ========================
# Main logic
# ========================
if __name__ == "__main__":
    # fpath = '../bag_recording/field_2021-07-16-11-46-47.bag'
    # fpath = '../bag_recording/field_2021-08-18-15-59-25.bag'
    # fpath      = '../bag_recording/field_2021-10-17-15-58-05.bag'
    bagname = sys.argv[1]
    fpath = "/afs/cs.stanford.edu/u/weizhuo2/Documents/Data_pipe/Bags/" + bagname
    # fpath = '/afs/cs.stanford.edu/u/weizhuo2/Documents/Data_pipe/Bags/realsense_2022-08-01-17-25-56_lag.bag'
    # fpath      = '../bag_recording/field_2021-12-09-16-45-58.bag.active'
    save_fpath = (
        "/afs/cs.stanford.edu/u/weizhuo2/Documents/Data_pipe/Training_sets/videos/"
        + fpath.split("/")[-1].split(".")[0]
    )  # extract the fname
    # video_channel = '/d400/color/image_raw'
    # # video_channel = '/testpano'

    # for video_channel in ["/testpano"]:
    # for video_channel in ['/d400/color/image_raw','/testpano']:
    for video_channel in ["/d400/color/image_raw"]:
        generator = video_generator(fpath)
        generator.collect_video(video_channel)
        generator.save_video(save_fpath)

    # Clean up
    plt.show()
    generator.exit()
