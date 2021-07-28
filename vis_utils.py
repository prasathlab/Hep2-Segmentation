import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from image_utils import *
import pdb

def get_imgs_from_names(query_df, path_df):
    #img_name_list = [os.path.basename(img_path) for img_path in path_df["imageNames"].tolist()]
    new_df = pd.DataFrame(index=query_df.index, columns=["imageNames", "maskNames"])
    for img_name, row in query_df.iterrows():
        match_bools = path_df["imageNames"].str.contains(img_name)
        temp_df = path_df[match_bools.values]
        assert (temp_df.shape[0] == 1)
        img_path = temp_df["imageNames"].tolist()[0]
        mask_path = temp_df["maskNames"].tolist()[0]
        new_df.loc[img_name, "imageNames"] = img_path
        new_df.loc[img_name, "maskNames"] = mask_path
    query_df = pd.merge(query_df, new_df, how='outer', left_index=True, right_index=True)
    return query_df

def do_contour_visualization(imgs_df, img_pred_dict, vis_dir, threshold_confusion=0.5):

    for img_name, row in imgs_df.iterrows():
        img_path = row["imageNames"]
        img_name_w_ext = os.path.basename(img_path)
        img_name = os.path.splitext(img_name_w_ext)[0]
        mask_path = row["maskNames"]
        js_score = row["Jaccard"]
        save_dir = os.path.join(vis_dir, img_name+f'_JS_{js_score:.3f}')
        os.makedirs(save_dir, exist_ok=True)
        #Load image
        test_img = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED)
        #Load mask
        test_mask = cv2.imread(mask_path, flags=cv2.IMREAD_UNCHANGED)
        test_mask = mask_binarization(test_mask)
        test_mask = test_mask*255
        #Get Prediction
        pred_mask = img_pred_dict[img_name_w_ext]
        pred_mask = np.squeeze(pred_mask)
        pred_mask = np.where(pred_mask > threshold_confusion, 255, 0)
        pred_mask = pred_mask.astype("uint8")

        #Save original_images, masks and predictions
        cv2.imwrite(os.path.join(save_dir, img_name+f"_JS_{js_score:.3f}.png"), test_img)
        cv2.imwrite(os.path.join(save_dir, img_name + f"_JS_{js_score:.3f}_GT_Mask.png"), test_mask)
        cv2.imwrite(os.path.join(save_dir, img_name + f"_JS_{js_score:.3f}_Pred_Mask.png"), pred_mask)

        #Get contours
        mask_contours, mask_hierarchy = cv2.findContours(test_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        print(f"Number of Contours in Mask = {len(mask_contours)}")
        pred_contours, pred_hierarchy = cv2.findContours(pred_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        print(f"Number of Contours in Prediction = {len(pred_contours)}")

        # Overlay mask and pred contours on blank image
        blank_img = np.zeros((test_mask.shape[0], test_mask.shape[1], 3))
        cv2.drawContours(blank_img, mask_contours, -1, (0, 255, 0), 3)
        cv2.drawContours(blank_img, pred_contours, -1, (255, 0, 0), 3)

        cv2.imwrite(os.path.join(save_dir, img_name+f'_JS_{js_score:.3f}_blank_cntrs.png'), blank_img)

        #Overlay mask and pred contours on the original image
        test_img_copy = np.copy(test_img)
        # Apply CLAHE to enhance the contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        test_img_copy = clahe.apply(test_img_copy)
        cv2.imwrite(os.path.join(save_dir, img_name + f"_JS_{js_score:.3f}_CLAHE.png"), test_img_copy)

        if test_img_copy.ndim < 3:
            test_img_copy = np.expand_dims(test_img_copy, axis=-1)
            test_img_copy = cv2.cvtColor(test_img_copy, cv2.COLOR_GRAY2RGB)

        cv2.drawContours(test_img_copy, mask_contours, -1, (0, 255, 0), 3)
        cv2.drawContours(test_img_copy, pred_contours, -1, (255, 0, 0), 3)
        cv2.imwrite(os.path.join(save_dir, img_name + f'_JS_{js_score:.3f}_img_cntrs.png'), test_img_copy)
