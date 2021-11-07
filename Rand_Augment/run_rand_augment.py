
import os
import cv2
from rand_augment import *
#from .file_utils import get_hep2_data
import numpy as np
import pandas as pd
import pdb

def get_hep2_img_mask(data_row):
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
        #img = np.expand_dims(img, axis=-1)
        #mask = np.expand_dims(mask, axis=-1)
    img = np.expand_dims(np.array(img), axis=-1)
    mask = np.expand_dims(np.array(mask), axis=-1)
    return img, mask

save_path = r"/data/aronow/Balaji_Iyer/Projects/Hep-2_Segmentation/Augmented_Data"
train_df_path = r"/data/aronow/Balaji_Iyer/Projects/Hep-2_Segmentation/Image_Classes/extended_train_data.tsv"
n_augs_per_img = 5
train_df = pd.read_csv(train_df_path, sep="\t", index_col=0)
train_aug_df = pd.DataFrame()
#imgs, masks = get_hep2_data(train_df)
augmenter = RandAugment(3, 8)

for idx, row in train_df.iterrows():
    img_name = row['img_short_name']
    mask_name = row['mask_short_name']
    pattern = row['pattern']
    img_id = row['ID']
    pattern_save_path = os.path.join(save_path, pattern)
    os.makedirs(pattern_save_path, exist_ok=True)
    img, mask = get_hep2_img_mask(row)
    temp_df = pd.DataFrame(columns=train_df.columns)
    print(f"image# = {idx}/{train_df.shape[0]}, img_name = {img_name}, pattern={pattern}")
    for j in range(n_augs_per_img):
        aug_img, aug_mask = augmenter(img, mask)
        aug_img_name = f"{img_name}_A{j}.tif"
        aug_mask_name = f"{mask_name}_A{j}.tif"
        aug_img_path = os.path.join(pattern_save_path, aug_img_name)
        aug_mask_path = os.path.join(pattern_save_path, aug_mask_name)
        cv2.imwrite(aug_img_path, aug_img)
        cv2.imwrite(aug_mask_path, aug_mask)
        temp_df.loc[j, "imageNames"] = aug_img_path
        temp_df.loc[j, "img_short_name"] = aug_img_name
        temp_df.loc[j, "maskNames"] = aug_mask_path
        temp_df.loc[j, "mask_short_name"] = aug_mask_name
        temp_df.loc[j, "pattern"] = pattern
        temp_df.loc[j, "ID"] = img_id
        temp_df.loc[j, "Aug_ID"] = j

    train_aug_df = pd.concat([train_aug_df, temp_df], axis="index")
    train_aug_df = train_aug_df.reset_index(drop=True)

train_aug_df.to_csv(os.path.join(save_path, "train_aug.tsv"), sep="\t")
pdb.set_trace()
debug = 1



#
# for idx, (img, mask) in enumerate(zip(imgs, masks)):
#     temp_df = pd.DataFrame(columns=train_df.columns)
#     for j in range(n_augs_per_img):
#         aug_img, aug_mask = augmenter(img, mask)
#         img_path = os.path.join(save_path, f"aug_image_{idx}_R{j}.tif")
#         mask_path = os.path.join(save_path, f"aug_mask_{idx}_R{j}.tif")
#         cv2.imwrite(img_path, aug_img)
#         cv2.imwrite(mask_path, aug_mask)
#         temp_df.loc[j, "imageNames"] = img_path
#         temp_df.loc[j, "maskNames"] = mask_path
#     train_aug_df = pd.concat([train_aug_df, temp_df], axis="index")
#     train_aug_df = train_aug_df.reset_index(drop=True)

# pdb.set_trace()
# train_aug_df.to_csv(os.path.join(save_path, "train_aug.tsv"), sep="\t")