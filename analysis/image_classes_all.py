import pandas as pd
import os
import numpy as np
import re


path_df = r"Z:\Balaji_Iyer\Projects\Hep-2_Segmentation\Image_Classes\new_all_paired_names.tsv"
paired_names_df = pd.read_csv(path_df, sep="\t", index_col=0)
paired_names_df = paired_names_df.reset_index(drop=True)

class_path = r"Z:\Balaji_Iyer\Projects\Hep-2_Segmentation\Image_Classes\class_specification.tsv"
class_df = pd.read_csv(class_path, sep="\t", index_col=0)

for idx, row in paired_names_df.iterrows():
	img_path = row["imageNames"]
	mask_path = row["maskNames"]
	img_basename = os.path.basename(img_path)
	img_short_name = os.path.splitext(img_basename)[0]
	mask_basename = os.path.basename(mask_path)
	mask_short_name = os.path.splitext(mask_basename)[0]
	paired_names_df.loc[idx, "img_short_name"] = img_short_name
	paired_names_df.loc[idx, "mask_short_name"] = mask_short_name
	img_id_split = re.split("_", img_short_name)
	img_id = img_id_split[0]
	img_id = int(img_id)
	match_row = class_df[class_df["ID"] == img_id]
	assert(match_row.shape[0]==1)
	pattern = match_row["pattern"].tolist()[0]
	paired_names_df.loc[idx, "ID"] = str(img_id)
	paired_names_df.loc[idx, "pattern"] = pattern

paired_names_df.to_csv(path_df, sep="\t")
