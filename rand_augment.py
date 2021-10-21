import cv2
import pandas as pd
from file_utils import get_hep2_data
from PIL import Image, ImageOps, ImageEnhance
import random
import numpy as np
import os
import pdb

#The code above is a modified version of https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py

def TranslateXabs(img, mask, v):
    #img and mask must be translated in exactly same way
    assert 0 <= v
    v = random.randint(0, 100)
    if random.random() > 0.5:
        v = -v
    img_transformed = img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))
    mask_transformed = mask.transform(mask.size, Image.AFFINE, (1, 0, v, 0, 1, 0))
    return img_transformed, mask_transformed


def TranslateYabs(img, mask, v):
    # img and mask must be translated in exactly same way
    assert 0 <= v
    v = random.randint(0, 100)
    if random.random() > 0.5:
        v = -v
    img_transformed = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))
    mask_transformed = mask.transform(mask.size, Image.AFFINE, (1, 0, 0, 0, 1, v))
    return img_transformed, mask_transformed


def Rotate(img, mask, v):
    # img and mask must be rotated in exactly same way
    assert -30 <= v <= 30
    v = random.randint(-130, 130)
    if random.random() > 0.5:
        v = -v
    img_transformed = img.rotate(v)
    mask_transformed = mask.rotate(v)
    return img_transformed, mask_transformed


def AutoContrast(img, mask, _):
    # img contrast can be changed. Mask must remain unchanged
    #https://pillow.readthedocs.io/en/stable/reference/ImageOps.html
    #cutoff: The percent to cut off from the histogram on the low and high ends.
            # Either a tuple of (low, high), or a single number for both.
    img_transformed = ImageOps.autocontrast(img, cutoff=(2,0))
    mask_transformed = mask
    return img_transformed, mask_transformed


def Equalize(img, mask, _):
    # img histogram can be equalized. Mask must remain unchanged
    img_transformed = ImageOps.equalize(img)
    mask_transformed = mask
    return img_transformed, mask_transformed


def Solarize(img, mask, v):
    # img can be solarized. Mask must remain unchanged
    assert 0 <= v <= 256
    v = random.randint(0, int(v))
    img_transformed = ImageOps.solarize(img, v)
    mask_transformed = mask
    return img_transformed, mask_transformed


def Posterize(img, mask, v):
    v = int(v)
    v = max(1, v)
    img_transformed = ImageOps.posterize(img, v)
    mask_transformed = mask
    return img_transformed, mask_transformed


def Contrast(img, mask, v):
    # img contrast can be changed. Mask must remain unchanged
    assert 0.1 <= v <= 1.9
    v = random.uniform(0.1, 1.9)
    img_transformed = ImageEnhance.Contrast(img).enhance(v)
    mask_transformed = mask
    return img_transformed, mask_transformed


def Color(img, mask, v):
    # img color can be changed. Mask must remain unchanged
    assert 0.1 <= v <= 1.9
    v = random.uniform(0.1, 1.9)
    img_transformed = ImageEnhance.Color(img).enhance(v)
    mask_transformed = mask
    return img_transformed, mask_transformed


def Brightness(img, mask, v):
    # img brightness can be changed. Mask must remain unchanged
    assert 0.1 <= v <= 1.9
    v = random.uniform(0.1, 1.9)
    img_transformed = ImageEnhance.Brightness(img).enhance(v)
    mask_transformed = mask
    return img_transformed, mask_transformed


def Sharpness(img, mask, v):
    assert 0.1 <= v <= 1.9
    v = random.uniform(0.1, 1.9)
    img_transformed = ImageEnhance.Sharpness(img).enhance(v)
    mask_transformed = mask
    return img_transformed, mask_transformed


def Identity(img, mask, _):
    return img, mask


class RandAugment:
    # n -> Number of augmentations per image
    # m ->
    def __init__(self, n, m, save_dir=None):
        self.save_dir = save_dir
        self.n = n
        self.m = m

        self.augmentations = [
            (Identity, 0, 1),
            (AutoContrast, 0, 1),
            (Rotate, 0, 30),
            (Equalize, 0, 1),
            (Posterize, 0, 1),
            (Solarize, 0, 256),
            #(Color, 0.1, 1.9),
            (Contrast, 0.3, 2.5),
            (Rotate, 0, 30),
            (Brightness, 0.3, 2.3),
            (Sharpness, 0.3, 2.3),
            (TranslateXabs, 0., 100),
            (TranslateYabs, 0., 100),
        ]

    def __call__(self, img, mask):
        #ops = random.choices(self.augmentations, k=self.n)
        ops = random.sample(self.augmentations, k=self.n)
        pil_img = Image.fromarray(np.squeeze(img, axis=2))
        pil_mask = Image.fromarray(np.squeeze(mask, axis=2))

        if self.save_dir:
            cv2.imwrite(os.path.join(self.save_dir, "img_orig.png"), np.array(pil_img))
            cv2.imwrite(os.path.join(self.save_dir,"mask_orig.png"), np.array(pil_mask))
        for op, minval, maxval in ops:
            print(op.__name__, end=", ")
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            pil_img, pil_mask = op(pil_img, pil_mask, val)
            # remove all lines below if not required
            if self.save_dir:
                cv2.imwrite(os.path.join(self.save_dir,f"img_{op.__name__}_{val}.png"), np.array(pil_img))
                cv2.imwrite(os.path.join(self.save_dir,f"mask_{op.__name__}_{val}.png"), np.array(pil_mask))
            #pil_img = Image.fromarray(np.squeeze(img, axis=2))
            #pil_mask = Image.fromarray(np.squeeze(mask, axis=2))
            ###########

        return (np.expand_dims(np.array(pil_img), 2), np.expand_dims(np.array(pil_mask), 2))

#TODO: Augs to disable: color

# save_path = r"/data/aronow/Balaji_Iyer/Projects/Hep-2_Segmentation/Augmented_Data"
# train_df_path = r"/data/aronow/Balaji_Iyer/Projects/Hep-2_Segmentation/raw_data/train_data.tsv"
# n_augs_per_img = 5
# pdb.set_trace()
# train_df = pd.read_csv(train_df_path, sep="\t", index_col=0)
# train_aug_df = pd.DataFrame(columns=train_df.columns)
# pdb.set_trace()
# imgs, masks = get_hep2_data(train_df)
# augmenter = RandAugment(3, 8)
# pdb.set_trace()
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
#
# pdb.set_trace()
# train_aug_df.to_csv(os.path.join(save_path, "train_aug.tsv"), sep="\t")
# pdb.set_trace()
# debug = 1




