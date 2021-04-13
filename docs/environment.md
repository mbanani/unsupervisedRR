# Setup 

This repository makes use of several external libraries. 
We highly recommend installing them within a virtual environment such as Anaconda. 

The script below will help you set up the environment; the `--yes` flag allows conda to install
without requesting your input for each package.

```bash 
conda create --name URR python=3.6
conda activate URR 

# pytorch 1.7.1 and torchvision 0.8.2
conda install pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=10.2 -c pytorch --yes

# matplotlib and tensorboard
conda install matplotlib tensorboard --yes

# update numpy 
conda install numpy==1.19.4 --yes

# PyTorch 0.4.0
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.4.0"

# Open3D 0.9 (Older version due to OS restrictions with RedHat)
pip install open3d==0.9

# MinkowskiEngine 0.5
conda install openblas-devel -c anaconda --yes
git clone -b v0.5.0 https://github.com/NVIDIA/MinkowskiEngine
cd MinkowskiEngine
python setup.py install --blas=openblas --blas_include_dirs=${CONDA_PREFIX}/include
cd ..

# other misc packages for baselines
pip install nibabel opencv-python easydict pre-commit  

# The following is not essential to run the code, but good if you want to contribute
# or just keep clean repositories. You should find a .pre-commit-config.yaml file 
# already in the repo.
cd <project_repo>
pre-commit install 
```
