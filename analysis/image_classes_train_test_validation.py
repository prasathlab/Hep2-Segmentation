import pandas as pd
import os
import numpy as np
import re


def populate_image_mask_short_names(data_df):
	for idx, row in data_df.iterrows():
		img_path = row["imageNames"]
		mask_path = row["maskNames"]
		img_basename = os.path.basename(img_path)
		img_short_name = os.path.splitext(img_basename)[0]
		mask_basename = os.path.basename(mask_path)
		mask_short_name = os.path.splitext(mask_basename)[0]
		data_df.loc[idx, "img_short_name"] = img_short_name
		data_df.loc[idx, "mask_short_name"] = mask_short_name
	return data_df


extended_path_all = r"Z:\Balaji_Iyer\Projects\Hep-2_Segmentation\Image_Classes\extended_all_paired_names.tsv"
extended_all_df = pd.read_csv(extended_path_all, sep="\t", index_col=0)
extended_all_df = extended_all_df.reset_index(drop=True)

extended_path_train = r"Z:\Balaji_Iyer\Projects\Hep-2_Segmentation\Image_Classes\extended_train_data.tsv"
extended_train_df = pd.read_csv(extended_path_train, sep="\t", index_col=0)
extended_train_df = extended_train_df.reset_index(drop=True)

extended_path_test = r"Z:\Balaji_Iyer\Projects\Hep-2_Segmentation\Image_Classes\extended_test_data.tsv"
extended_test_df = pd.read_csv(extended_path_test, sep="\t", index_col=0)
extended_test_df = extended_test_df.reset_index(drop=True)

extended_path_val = r"Z:\Balaji_Iyer\Projects\Hep-2_Segmentation\Image_Classes\extended_val_data.tsv"
extended_val_df = pd.read_csv(extended_path_val, sep="\t", index_col=0)
extended_val_df = extended_val_df.reset_index(drop=True)

extended_train_df = populate_image_mask_short_names(extended_train_df)
extended_test_df = populate_image_mask_short_names(extended_test_df)
extended_val_df = populate_image_mask_short_names(extended_val_df)


extended_train_df = pd.merge(extended_train_df, extended_all_df,
                             on=['imageNames','maskNames', 'img_short_name', 'mask_short_name'],
                             how='inner'
                             )
extended_test_df = pd.merge(extended_test_df, extended_all_df,
                             on=['imageNames','maskNames', 'img_short_name', 'mask_short_name'],
                             how='inner'
                             )

extended_val_df = pd.merge(extended_val_df, extended_all_df,
                             on=['imageNames','maskNames', 'img_short_name', 'mask_short_name'],
                             how='inner'
                             )

extended_train_df.to_csv(extended_path_train, sep="\t")
extended_test_df.to_csv(extended_path_test, sep="\t")
extended_val_df.to_csv(extended_path_val, sep="\t")
debug = 1