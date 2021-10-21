import numpy as np
import random
import os
import pdb
from rand_augment import *
from image_utils import *
import cv2

def img_convert(img, target_type_min, target_type_max, target_type):
    img_min = img.min()
    img_max = img.max()
    print(img_max)
    print(img_min)
    if img_max - img_min <=0:
        #pdb.set_trace()
        print("Here it is> Divide by Zero")

    a = (target_type_max - target_type_min) / (img_max - img_min)
    b = target_type_max - a * img_max
    new_img = (a * img + b).astype(target_type)
    return img_min, img_max, new_img

def get_batch(img_files, mask_dir, batch_size):
    sample_idxs = random.sample(range(len(img_files)), k=batch_size)
    batch_imgs, batch_masks = [], []
    for idx in sample_idxs:
        img = np.load(img_files[idx])
        batch_imgs.append(img)
        img_basename = os.path.basename(img_files[idx])
        img_name = os.path.splitext(img_basename)[0]
        mask_name = img_name + "_mask.npy"
        mask_path = os.path.join(mask_dir, mask_name)
        mask = np.load(mask_path)
        batch_masks.append(mask)
    batch_imgs = np.array(batch_imgs)
    batch_masks = np.array(batch_masks)
    check = batch_imgs.shape == batch_masks.shape
    return batch_imgs, batch_masks


def make_generator_w_aug(img_files, mask_dir, batch_size, aug_img_files, aug_mask_dir):
    unaug_batch_size = 1
    #unaug_batch_size = int(batch_size//2)
    aug_batch_size = batch_size - unaug_batch_size
    while 1:
        unaug_batch_imgs, unaug_batch_masks = get_batch(img_files, mask_dir, unaug_batch_size)
        aug_batch_imgs, aug_batch_masks = get_batch(aug_img_files, aug_mask_dir, aug_batch_size)
        batch_imgs = np.concatenate([unaug_batch_imgs, aug_batch_imgs], axis=0)
        batch_masks = np.concatenate([unaug_batch_masks, aug_batch_masks], axis=0)
        yield (batch_imgs, batch_masks)


def make_generator(img_files, mask_dir, batch_size ):
    while 1:
        batch_imgs, batch_masks = get_batch(img_files, mask_dir, batch_size)
        yield (batch_imgs, batch_masks)

def split_train_val_set(img_files, valid_ratio):
    n_val_samples = int(valid_ratio*len(img_files))
    val_img_files = random.sample(img_files, k=n_val_samples)
    train_img_files = [img_file for img_file in img_files if img_file not in val_img_files]
    check = [x for x in train_img_files if x in val_img_files]
    return train_img_files, val_img_files

def get_validation_data(val_img_files, mask_dir):
    val_imgs, val_masks  = [], []
    for val_img_file in val_img_files:
        val_img = np.load(val_img_file)
        val_imgs.append(val_img)
        img_basename = os.path.basename(val_img_file)
        img_name = os.path.splitext(img_basename)[0]
        mask_name = img_name + "_mask.npy"
        mask_path = os.path.join(mask_dir, mask_name)
        val_mask = np.load(mask_path)
        val_masks.append(val_mask)
    val_imgs = np.array(val_imgs)
    val_masks = np.array(val_masks)
    return val_imgs, val_masks


