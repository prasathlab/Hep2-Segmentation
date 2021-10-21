import os
import cv2
import pandas as pd
from rand_augment import *
import pdb






save_path = r"/data/aronow/Balaji_Iyer/Projects/Hep-2_Segmentation/Augmented_Data"
train_df_path = r"/data/aronow/Balaji_Iyer/Projects/Hep-2_Segmentation/raw_data/train_data.tsv"
n_augs_per_img = 5
pdb.set_trace()
train_df = pd.read_csv(train_df_path, sep="\t", index_col=0)
train_aug_df = pd.DataFrame(columns=train_df.columns)
pdb.set_trace()
imgs, masks = get_hep2_data(train_df)
augmenter = RandAugment(3, 8)
pdb.set_trace()
for idx, (img, mask) in enumerate(zip(imgs, masks)):
    temp_df = pd.DataFrame(columns=train_df.columns)
    for j in range(n_augs_per_img):
        aug_img, aug_mask = augmenter(img, mask)
        img_path = os.path.join(save_path, f"aug_image_{idx}_R{j}.tif")
        mask_path = os.path.join(save_path, f"aug_mask_{idx}_R{j}.tif")
        cv2.imwrite(img_path, aug_img)
        cv2.imwrite(mask_path, aug_mask)
        temp_df.loc[j, "imageNames"] = img_path
        temp_df.loc[j, "maskNames"] = mask_path
    train_aug_df = pd.concat([train_aug_df, temp_df], axis="index")
    train_aug_df = train_aug_df.reset_index(drop=True)

pdb.set_trace()
train_aug_df.to_csv(os.path.join(save_path, "train_aug.tsv"), sep="\t")