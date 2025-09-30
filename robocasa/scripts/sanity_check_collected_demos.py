import argparse
from copy import deepcopy
import datetime
import json
import os
import time
from glob import glob

import h5py
import imageio
import mujoco
import numpy as np
import robosuite
import pdb


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo_path",
        type=str,
    )
    args = parser.parse_args()
    demo_path = args.demo_path

    f = h5py.File(demo_path, "r")  # read mode

    # Check the contents of the HDF5 file
    print("Keys in HDF5 file:")
    for key in f.keys():
        print(f" - {key}")
    print("\nAttributes in 'data' group:")
    for attr in f["data"].attrs:
        print(f" - {attr}: {f['data'].attrs[attr]}")

    # plot the distribution of actions as a histogram
    all_actions = []
    for key in f["data"].keys():
        actions = f["data"][key]["actions"][:][0]
        all_actions.extend(actions)
    # all_actions = np.concatenate(all_actions, axis=0)
    # pdb.set_trace()
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(all_actions, bins=50)
    plt.title("Distribution of Actions")
    plt.xlabel("Action Value")
    plt.ylabel("Frequency")
    plt.show()

    # Plot each trajectory as video in a new folder in the current directory
    # pdb.set_trace()
    timestamp_for_demo = demo_path.split("/")[-2]
    foldername = "data_sanity_check"
    # create in current directory if foldername does not exist
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    foldername = os.path.join(foldername, timestamp_for_demo)
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    for key in f["data"].keys():
        # get keys in f['data'][key]['obs'] with image in the name
        image_keys = [k for k in f["data"][key]["obs"].keys() if "image" in k]
        if len(image_keys) == 0:
            print(f"No image keys found in trajectory {key}, skipping...")
            continue
        # get all image keys, and for each timestep horizontally stack the images
        images = []
        for img_key in image_keys:
            img = f["data"][key]["obs"][img_key][:]
            images.append(img)
        # horizontally stack the images for each timestep
        stacked_images = [
            np.hstack([images[i][t] for i in range(len(images))])
            for t in range(images[0].shape[0])
        ]
        video_path = os.path.join(foldername, f"traj_{key}.mp4")
        imageio.mimwrite(video_path, stacked_images, fps=30)
        print(f"Saved video for trajectory {key} at {video_path}")
