# Datasets 

We use two datasets in this work: ScanNet and 3D Match. For each dataset, you have to download the
dataset directly from the authors and following the instructions below for creating a data
dictionary for the data loaders. The data dictionary allows us to easily interface with the dataset
by defining paths in a consistent way as well as pre-extracting the intrinsic and pose matrices. 
We emphasize that the pose matrices are only used for evaluation. 

Once you download the datasets, make sure to update the paths in
`unsuperviserRR/datasets/builder.py` with the dataset root directories. 

## ScanNet

ScanNet is a large dataset of indoor scenes: over 1500 scenes and 2.5 million views. 
The dataset is organized as a series of RGB-D sequences that are stored as a sensor-stream (or
.sens) file. Below are the instructions to download, extract, and process the dataset as well as
generate the data dictionaries that are used by the data loaders to index into the dataset. 

You first need to download the dataset. This can be done by following the instructions outlined in
the official [ScanNet repository](https://github.com/ScanNet/ScanNet). 
We downloaded the individual scenes and used the official splits found
[here](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark). 
We use the v1 splits in this work.  

Once all the scenes are downloaded and extracted, we processed the `.sens` files used the provided
[SensReader](https://github.com/ScanNet/ScanNet/tree/master/SensReader/python) to extract the color,
depth, intrinsics, and pose matrices. We note that the pose matrices are only used for evaluation. 

Finally, you can use `data/create_scannet_dict.py` to generate data dictionaries for each of the
splits. The script assumes the following directory structure for ScanNet

```
<ScanNet Root>
    |- train
    |- valid
    |- test
        |- scans
            |- scene0024_01 
            |- scene0024_02 
            |- scene0024_03 
                |- color 
                    |- 0.jpg
                    |- 1.jpg
                    |- 2.jpg
                |- depth 
                    |- 0.png
                    |- 1.png
                    |- 2.png
                |- pose
                    |- 0.txt
                    |- 1.txt
                    |- 2.txt
                |- intrinsic
                    |- intrinsic_color.txt
                    |- intrinsic_depth.txt
                    |- extrinsic_color.txt
                    |- extrinsic_depth.txt
```

If this structure is followed, you can simply run the script to generate the data dictionary 
for each split as follows:

```
cd data 
python create_scannet_dict.py <ScanNet Root> scannet_train.pkl train
python create_scannet_dict.py <ScanNet Root> scannet_valid.pkl valid
python create_scannet_dict.py <ScanNet Root> scannet_test.pkl test
```

Please note that this script's speed is highly dependant on the disk IO speed; 
it can be very slow if you're running on a compute cluster with NFS storage.

## 3DMatch

[3DMatch](https://3dmatch.cs.princeton.edu) is a dataset that has been used to benchmark several 3D
vision tasks. In this work, we use the sequences from the RGB-D reconstruction benchmark. The
dataset consists of several other smaller RGB-D sequence datasets. 

You can use the download
[script](http://vision.princeton.edu/projects/2016/3DMatch/downloads/rgbd-datasets/download.sh) 
provided by 3D Match to download all the sequences as zip files. You then need to extract each of
those sequences in the same directory. 

We use the [official
split](http://vision.princeton.edu/projects/2016/3DMatch/downloads/rgbd-datasets/split.txt), but we
further split the training set into train and valid. The final scene level split can be seen in 
`data/3dmatch_splits.txt`.  

Finally, you can use `data/create_3dmatch_rgbd_dict.py` to generate data dictionaries for each of the
splits. You can simply run the script to generate the data dictionary 
for each split as follows:
```
cd data 
python create_3dmatch_rgbd_dict.py <3D Match Root> 3dmatch_train.pkl train
python create_3dmatch_rgbd_dict.py <3D Match Root> 3dmatch_valid.pkl valid
python create_3dmatch_rgbd_dict.py <3D Match Root> 3dmatch_test.pkl test
```

