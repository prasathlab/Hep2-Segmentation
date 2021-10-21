import os
import h5py
import numpy as np
import pandas as pd
from PIL import Image
import yaml
import wandb
import glob
import cv2
from file_utils import *
from image_utils import *
import random
import matplotlib.pyplot as plt
import psutil
import pdb


yaml_file = r"/data/aronow/Balaji_Iyer/Projects/Hep-2_Segmentation/Hep2-Segmentation/prepare_data.yaml"

with open(yaml_file, 'r') as stream:
    try:
        cfg = yaml.load(stream)
        print(cfg)
    except yaml.YAMLError as exc:
        print(exc)


#Read parameters
patch_dim = cfg["prepare_data"]["patch_dim"]
normalize = cfg["prepare_data"]["normalize"]
n_patches_per_img = cfg["prepare_data"]["n_patches_per_img"]
save_path = cfg["prepare_data"]["save_path"]
save_dir = os.path.join(save_path, f"aug_data_npy_D_{patch_dim[0]}")
os.makedirs(save_dir, exist_ok=True)

#Create dirs for saving train data
train_img_save_dir = os.path.join(save_dir, "Train_Images")
os.makedirs(train_img_save_dir, exist_ok=True)
train_mask_save_dir = os.path.join(save_dir, "Train_Masks")
os.makedirs(train_mask_save_dir, exist_ok=True)

#Create dirs for saving val data
# val_img_save_dir = os.path.join(save_dir, "Val_Images")
# os.makedirs(val_img_save_dir, exist_ok=True)
# val_mask_save_dir = os.path.join(save_dir, "Val_Masks")
# os.makedirs(val_mask_save_dir, exist_ok=True)

#Parse and Extract patches from Training Images

train_df = pd.read_csv(cfg["prepare_data"]["train_data"], sep="\t", index_col=0)
train_imgs, train_masks = get_hep2_data(train_df)
#Normalize the images before extracting patches

train_imgs, train_masks = image_mask_scaling(train_imgs, train_masks)
print(f"After scaling: train_imgs(max={np.amax(train_imgs)}, min={np.amin(train_imgs)})")
print(f"After scaling: train_masks(max={np.amax(train_masks)}, min={np.amin(train_masks)})")


if not cfg["prepare_data"]["normalization_params"]:

    train_imgs, train_img_ch_mean, train_img_ch_std = get_normalization_params(train_imgs)
    train_img_ch_mean = np.expand_dims(np.squeeze(train_img_ch_mean), axis=0)
    train_img_ch_std = np.expand_dims(np.squeeze(train_img_ch_std), axis=0)
    img_params = np.concatenate([train_img_ch_mean, train_img_ch_std], axis=0)
    img_params_df = pd.DataFrame(img_params, columns=["GrayScale"], index=["Mean", "Std"])
    img_params_df.to_csv(os.path.join(save_path, "image_params.tsv"), sep="\t")

else:

    norm_params_df = pd.read_csv(cfg["prepare_data"]["normalization_params"], sep="\t", index_col=0)
    train_img_ch_mean = norm_params_df.loc["Mean", :].to_numpy()
    train_img_ch_std = norm_params_df.loc["Std", :].to_numpy()


if normalize:

    train_imgs = (train_imgs - train_img_ch_mean) / train_img_ch_std

train_img_patches, train_mask_patches = get_random_patches(train_imgs, train_masks, patch_dim, n_patches_per_img)


#save train_img_patches, train_mask_patches
if cfg["prepare_data"]["save_as_npy"]:
    save_images_masks(train_img_patches, train_mask_patches, train_img_save_dir, train_mask_save_dir)
    np.save(os.path.join(save_dir, f'train_imgs_patches_D_{patch_dim[0]}'), train_img_patches)
    np.save(os.path.join(save_dir, f'train_masks_patches_D_{patch_dim[0]}'), train_mask_patches)



#Parse and Extract patches from Validation Images

val_df = pd.read_csv(cfg["prepare_data"]["val_data"], sep="\t", index_col=0)
val_imgs, val_masks = get_hep2_data(val_df)
#Normalize the images before extracting patches

val_imgs, val_masks = image_mask_scaling(val_imgs, val_masks)

print(f"After scaling: val_imgs(max={np.amax(val_imgs)}, min={np.amin(val_imgs)})")
print(f"After scaling: val_masks(max={np.amax(val_masks)}, min={np.amin(val_masks)})")

val_imgs = (val_imgs - train_img_ch_mean) / train_img_ch_std
val_img_patches, val_mask_patches = get_random_patches(val_imgs, val_masks, patch_dim, n_patches_per_img)

if cfg["prepare_data"]["save_as_npy"]:
    save_images_masks(val_img_patches, val_mask_patches, val_img_save_dir, val_mask_save_dir)
    #save full npy files as well. Just in case required
    np.save(os.path.join(save_dir, f'val_imgs_patches_D_{patch_dim[0]}'), val_img_patches)
    np.save(os.path.join(save_dir, f'val_masks_patches_D_{patch_dim[0]}'), val_mask_patches)





