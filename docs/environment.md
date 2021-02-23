# Setup 

This repository makes use of several external libraries. 
We highly recommend installing them within a virtual environment such as Anaconda. 

The script below will help you set up the environment; the `--yes` flag allows conda to install
without requesting your input for each package.

```bash 
conda create --name UnsupRR python=3.6
conda activate UnsupRR 

# pytorch 1.7.1 and torchvision 0.8.2
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch --yes

# matplotlib and tensorboard
conda install matplotlib tensorboard --yes

# pytorch3d-0.3.0
pip3 install "git+https://github.com/facebookresearch/pytorch3d.git@v0.3.0"

# Open3D 0.9 (Older version due to OS restrictions with RedHat)
pip3 install open3d==0.9

# MinkowskiEngine 0.5 (for baselines) -- make sure GCC > 7.4
# Note that there are some bugs with MinkowskiEngine and torch compiled with cuda 11.0
conda install openblas-devel -c anaconda --yes
git clone https://github.com/NVIDIA/MinkowskiEngine
cd MinkowskiEngine
python3 setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
cd ..

# other misc packages for baselines
pip3 install nibabel opencv-python 

# Open3D 0.9 (Older version due to OS restrictions with RedHat)
# other misc packages for baselines
# pre-commit helps keep repos clean 
pip3 install open3d==0.9 nibabel opencv-python easydict pre-commit

# The following is not essential to run the code, but good if you want to contribute
# or just keep clean repositories. You should find a .pre-commit-config.yaml file 
# already in the repo.
cd <project_repo>
pre-commit install 
```


