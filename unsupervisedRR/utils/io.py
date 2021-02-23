"""
Some useful I/O functions
"""
import os
import pickle
import shutil


# get all directories in a specific directory
def get_directories(path):
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]


# get all the files in a specific directory
# extension can be string or tuple of strings
def get_files(path, extension=None):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if extension is not None:
        files = [f for f in files if f.lower().endswith(extension)]
    return files


# get all files in a specific directory
def file_exists(path):
    return not os.path.exists(path)


# make directory
def makedir(path, replace_existing=False):
    if not os.path.exists(path):
        os.makedirs(path)
    elif replace_existing:
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        print("Beware .. path {} already exists".format(path))


# extract relative path from a root-directory and an absolute path
def relative_path(root, path):
    return os.path.relpath(path, root)


# save pickle
def save_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


# load pickle
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
