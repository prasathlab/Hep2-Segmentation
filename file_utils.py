import os
import h5py
import numpy as np
import pandas as pd
from PIL import Image
import datetime
import cv2
import argparse
import random
from sklearn.model_selection import train_test_split
import pdb


def save_images_masks(imgs, masks, img_save_dir, mask_save_dir):
    for idx, (img, mask) in enumerate(zip(imgs, masks)):
        if idx % 100 == 0:
            print(idx)
        np.save(os.path.join(img_save_dir, f"img_{idx}"), img)
        np.save(os.path.join(mask_save_dir, f"img_{idx}_mask"), mask)


def get_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False, description='Get inputs for Classification Pipeline')
    # Add the arguments
    parser.add_argument('--base_dir',
                        dest='base_dir',
                        required=True,
                        type=str,
                        help='Base Directory Path')

    parser.add_argument('--train_dir',
                        dest='train_dir',
                        required=True,
                        type=str,
                        help='Train Directory Name')

    parser.add_argument('--model_dir',
                        dest='model_dir',
                        required=True,
                        type=str,
                        help='Model Directory Name')

    parser.add_argument('--tb_dir',
                        dest='tb_dir',
                        required=True,
                        type=str,
                        help='Tensorboard Directory Name')

    parser.add_argument('--final_model_dir',
                        dest='final_model_dir',
                        required=True,
                        type=str,
                        help='Final Model Directory Name')

    return parser

def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)

def get_image_size(img_files):
    shape_df = pd.DataFrame(columns=["ImageName", "width", "height", "channels"])
    for idx, img_file in enumerate(img_files):
        img = np.asarray(Image.open(img_file))
        img_basename = os.path.basename(img_file)
        shape_df.loc[idx, "ImageName"] = img_basename
        shape_df.loc[idx, 1:] = img.shape
        if idx%100 == 0:
            print(idx)
    return shape_df


def pair_image_mask_names(all_files):
    paired_df = pd.DataFrame(columns=["imageNames", "maskNames"])
    for idx, filename in enumerate(all_files):
        if filename.endswith("_Mask.tif"):
            continue;
        else:
            img_name = filename
            mask_name = filename.replace(".tif", "") + "_Mask.tif"
            paired_df.loc[idx, "imageNames"] = img_name
            paired_df.loc[idx, "maskNames"] = mask_name

    paired_df.reset_index(drop=True)
    return (paired_df)

def get_hep2_data(data_df):
    pdb.set_trace()
    imgs = []
    masks = []
    idx = 0
    for _, data_row in data_df.iterrows():
        img_path = data_row["imageNames"]
        mask_path = data_row["maskNames"]
        img = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, flags=cv2.IMREAD_UNCHANGED)
        if len(img.shape) != 2:
            #image: 00252_p2.tif shape=(1040, 1388,4).
            #this is weird and unexpected, hence, changing it to grayscale
            print(f"img.shape = {img.shape}, mask.shape = {mask.shape}")
            img = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
            print(f"img.shape = {img.shape}, mask.shape = {mask.shape}")
        img = np.expand_dims(img, axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        imgs.append(img)
        masks.append(mask)
        if idx % 100 == 0:
            print(f"{idx} of {data_df.shape[0]}")
        idx += 1
    pdb.set_trace()
    return np.array(imgs), np.array(masks)

def setup_results_dir(res_dir="./Results", tb_dir="tb_log", time_stamp=True, ):
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_dir = os.path.join(res_dir, time_str)
    train_dir = os.path.join(res_dir, time_str, "Train")
    os.makedirs(train_dir, exist_ok=True)
    test_dir = os.path.join(res_dir, time_str, "Test")
    os.makedirs(test_dir, exist_ok=True)
    tb_dir = os.path.join(res_dir, time_str, tb_dir)
    os.makedirs(res_dir, exist_ok=True)
    mdl_dir = os.path.join(res_dir, time_str, "Models")
    os.makedirs(mdl_dir, exist_ok=True)
    return (exp_dir, train_dir, test_dir, tb_dir, mdl_dir)

def export_model_structure(model, file_location):
    plot_model(
        model=model,
        show_shapes=True,
        show_layer_names=True,
        to_file=file_location
        )