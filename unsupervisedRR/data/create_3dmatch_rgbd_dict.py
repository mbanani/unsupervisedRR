"""
Create a 3DMatch_RGBD dataset for the raw scans data
"""
import argparse
import os
import pickle

import numpy as np
from tqdm import tqdm


def get_splits():
    with open("3dmatch_splits.txt", "r") as f:
        splits = f.readlines()

    splits = [split.strip().split(", ") for split in splits]
    train = [split[1] for split in splits if split[0] == "TRAIN"]
    valid = [split[1] for split in splits if split[0] == "VALID"]
    test = [split[1] for split in splits if split[0] == "TEST"]
    return train, valid, test


def create_3dmatch_dict(data_root, dict_path, scenes):
    class_id = "3DMatch_RGBD"
    data_dict = {}

    data_dict = {class_id: {}}

    all_sseq_pairs = []
    # count num of sseqs
    for s_id in tqdm(scenes):
        sequences = os.listdir(os.path.join(data_root, s_id))
        sequences = [seq for seq in sequences if "seq-" in seq]

        for seq in sequences:
            all_sseq_pairs.append((s_id, seq))

    print(
        f"{len(scenes)} scenes -- total of {len(all_sseq_pairs)} Scene-Sequence pairs"
    )

    for i, sseq_pair in enumerate(all_sseq_pairs):
        s_id, seq = sseq_pair

        # get camera intrinsics
        int_path = os.path.join(data_root, s_id, "camera-intrinsics.txt")
        int_mat = np.loadtxt(int_path)

        sseq = f"{s_id}_{seq}"
        print(f"Processing sseq {sseq} ({i}/{len(all_sseq_pairs)})")

        data_dict[class_id][sseq] = {"instances": {}}
        sseq_path = os.path.join(data_root, s_id, seq)

        # get frames
        seq_files = os.listdir(sseq_path)
        seq_files = [name for name in seq_files if ".color.png" in name]
        seq_files = [name.split(".color")[0] for name in seq_files]
        seq_files.sort()

        inst_dict = {}
        skipped = 0
        for seq_file in tqdm(seq_files):
            # get relative paths
            rgb_rel = os.path.join(s_id, seq, f"{seq_file}.color.png")
            dep_rel = os.path.join(s_id, seq, f"{seq_file}.depth.png")
            ext_rel = os.path.join(s_id, seq, f"{seq_file}.pose.txt")

            # check for existance
            rgb_abs = os.path.join(data_root, rgb_rel)
            dep_abs = os.path.join(data_root, dep_rel)
            ext_abs = os.path.join(data_root, ext_rel)

            rgb_exist = os.path.exists(rgb_abs)
            dep_exist = os.path.exists(dep_abs)
            ext_exist = os.path.exists(ext_abs)

            if not (rgb_exist and dep_exist and ext_exist):
                skipped += 1
                continue

            # get values
            ext_file = os.path.join(data_root, ext_rel)
            ext_mat = np.loadtxt(ext_file)
            if np.isinf(ext_mat).any():
                skipped += 1
                continue

            # populate dictionary
            frame_id = int(seq_file.split("-")[1])
            inst_dict[frame_id] = {
                "rgb_path": rgb_rel,
                "dep_path": dep_rel,
                "extrinsic": ext_mat,
                "intrinsic": int_mat,
            }
        if skipped > 0:
            total = len(seq_files)
            print(f"Skipped {skipped}/{total} in {sseq}: inf P or missing files")
        data_dict[class_id][sseq]["instances"] = inst_dict

    # save dictionary as pickle in output path
    with open(dict_path, "wb") as f:
        pickle.dump(data_dict, f, protocol=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument("dict_path", type=str)
    parser.add_argument("split", type=str)
    args = parser.parse_args()

    train, valid, test = get_splits()
    if args.split == "train":
        out_split = train
    elif args.split == "valid":
        out_split = valid
    elif args.split == "test":
        out_split = test

    create_3dmatch_dict(args.data_root, args.dict_path, out_split)
