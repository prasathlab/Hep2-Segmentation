import cv2
import pandas as pd
from file_utils import get_hep2_data
from PIL import Image, ImageOps, ImageEnhance
import random
import numpy as np

#The code above is a modified version of https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py

def TranslateXabs(img, v):
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateYabs(img, v):
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return ImageOps.autocontrast(img)


def Equalize(img, _):
    return ImageOps.equalize(img)


def Solarize(img, v):
    assert 0 <= v <= 256
    return ImageOps.solarize(img, v)


def Posterize(img, v):
    v = int(v)
    v = max(1, v)
    return ImageOps.posterize(img, v)


def Contrast(img, v):
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Sharpness(img).enhance(v)


def Identity(img, _):
    return img


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.augmentations = [
            (Identity, 0, 1),
            (AutoContrast, 0, 1),
            (Equalize, 0, 1),
            (Rotate, 0, 30),
            (Posterize, 0, 4),
            (Solarize, 0, 256),
            (Color, 0.1, 1.9),
            (Contrast, 0.1, 1.9),
            (Brightness, 0.1, 1.9),
            (Sharpness, 0.1, 1.9),
            (TranslateXabs, 0., 100),
            (TranslateYabs, 0., 100),
        ]

    def __call__(self, img, mask):
        ops = random.choices(self.augmentations, k=self.n)
        pil_img = Image.fromarray(np.squeeze(img, axis=2))
        pil_mask = Image.fromarray(np.squeeze(mask, axis=2))

        for op, minval, maxval in ops:
            print(op, end=", ")
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            pil_img = op(pil_img, val)
            pil_mask = op(pil_mask, val)

        return (np.expand_dims(np.array(pil_img), 2), np.expand_dims(np.array(pil_mask), 2))


# train_df = pd.read_csv("train_data.tsv", sep="\t", index_col=0)
# r = RandAugment(3, 8)
#
# counter = 0
# imgs, masks = get_hep2_data(train_df)
# for i in zip(imgs, masks):
#     img, mask = r(i[0], i[1])
#     cv2.imwrite(f"augmented_images/{counter}.tif", img)
#     cv2.imwrite(f"augmented_images/{counter}_Mask.tif", mask)
#     counter += 1


