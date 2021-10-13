import numpy as np
import random
import os
import pdb
from rand_augment import *
import cv2

#instance of the RandAugment class
def make_generator(img_files, mask_dir, batch_size, augment=False):
    if augment:
        batch_size = int(batch_size//2)
    while 1:
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
        if augment:
            pdb.set_trace()
            aug_imgs = np.zeros_like(batch_imgs)
            aug_masks = np.zeros_like(batch_masks)
            rand_aug = RandAugment(3, 8)
            for idx, (img, mask) in enumerate(zip(batch_imgs, batch_masks)):
                aug_img, aug_mask = rand_aug(img, mask)
                aug_imgs[idx, :, :, :] = aug_img
                aug_masks[idx, :, :, :] = aug_mask
            batch_imgs = np.concatenate([batch_imgs, aug_imgs], axis=-1)
            batch_masks = np.concatenate([batch_masks, aug_imgs], axis=-1)
            pdb.set_trace()
        return (batch_imgs, batch_masks)

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


