"""
Create split-specific data dictionaries for ScanNet.
"""
import argparse
import os
import pickle

import numpy as np
from tqdm import tqdm


def create_scannet_dict(data_root, dict_path, split):
    class_id = "ScanNet"
    data_dict = {}

    scans_root = os.path.join(data_root, split, "scans")
    scenes = os.listdir(scans_root)
    data_dict = {class_id: {}}
    for s_id in tqdm(scenes):
        if "scene" not in s_id:
            print("Skipping {} as it doesn't contain scene".format(s_id))
            continue

        s_id = s_id.strip()
        data_dict[class_id][s_id] = {"instances": {}}

        # get relative paths
        rgb_rel = os.path.join(split, "scans", s_id, "color")
        dep_rel = os.path.join(split, "scans", s_id, "depth")
        ext_rel = os.path.join(split, "scans", s_id, "pose")

        # -- Get frames through ext_files --
        rgb_dir = os.path.join(data_root, rgb_rel)
        rgb_files = os.listdir(rgb_dir)

        # -- get intrisnics --
        int_path = os.path.join(
            data_root, split, "scans", s_id, "intrinsic/intrinsic_color.txt"
        )
        int_mat = np.loadtxt(int_path)
        inst_dict = {}
        skipped = 0
        for f_id in rgb_files:
            # get frame id from {frame_id}.jpg
            f_id = int(f_id.split(".")[0])

            # get values
            ext_file = os.path.join(data_root, ext_rel, "{}.txt".format(f_id))
            ext_mat = np.loadtxt(ext_file)
            if np.isinf(ext_mat).any():
                skipped += 1
                continue

            # populate dictionary
            inst_dict[f_id] = {
                "rgb_path": os.path.join(rgb_rel, "{}.jpg".format(f_id)),
                "dep_path": os.path.join(dep_rel, "{}.png".format(f_id)),
                "extrinsic": ext_mat,
                "intrinsic": int_mat,
            }
        if skipped > 0:
            print(f"Skipped {skipped}/{len(rgb_files)} frames for scene {s_id}")
        data_dict[class_id][s_id]["instances"] = inst_dict

    # save dictionary as pickle in output path
    with open(dict_path, "wb") as f:
        pickle.dump(data_dict, f, protocol=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument("dict_path", type=str)
    parser.add_argument("split", type=str)
    args = parser.parse_args()

    create_scannet_dict(args.data_root, args.dict_path, args.split)
