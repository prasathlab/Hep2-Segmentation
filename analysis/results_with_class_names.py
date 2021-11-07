import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

results_dir_path = r"Z:\Balaji_Iyer\Projects\Hep-2_Segmentation\Results\New_Models"
res_dirs = [name for name in os.listdir(results_dir_path) if os.path.isdir(os.path.join(results_dir_path, name))]

all_1008_dir = "All_1008"
test_res_dir = "Test"

names_path = r"Z:\Balaji_Iyer\Projects\Hep-2_Segmentation\Image_Classes\extended_all_paired_names.tsv"
paired_names_df = pd.read_csv(names_path, sep="\t", index_col=0)
paired_grouped = paired_names_df.groupby(by="pattern")

#model_path = r"Z:\Balaji_Iyer\Projects\Hep-2_Segmentation\Results\New_Models\fcn_8_resnet50_NPT_FT_2021-09-11_15-30-36"


for res_dir in res_dirs:
	model_path = os.path.join(results_dir_path, res_dir)
	test_path = os.path.join(model_path, test_res_dir, "image_performances.tsv")
	test_df = pd.read_csv(test_path, sep="\t", index_col=0)
	test_img_names = test_df.index.tolist()
	assert(len(set(test_img_names)) == len(test_img_names))

	all_1008_path = os.path.join(model_path, all_1008_dir, "image_performances.tsv")
	all_1008_df = pd.read_csv(all_1008_path, sep="\t", index_col=0)
	all_1008_img_names = all_1008_df.index.tolist()
	assert(len(set(all_1008_img_names)) == len(all_1008_img_names))

	for pattern, group_df in paired_grouped:
		img_names = group_df.loc[:, "img_short_name"].tolist()
		img_names_w_ext = [name + ".tif" for name in img_names]

		all_img_names_w_ext = list(set(img_names_w_ext).intersection(set(all_1008_img_names)))
		all_1008_df.loc[all_img_names_w_ext, "pattern"] = pattern

		test_img_names_w_ext = list(set(img_names_w_ext).intersection(set(test_img_names)))
		test_df.loc[test_img_names_w_ext, "pattern"] = pattern

	test_df.to_csv(test_path, sep="\t")
	all_1008_df.to_csv(all_1008_path, sep="\t")

	debug = 1

debug = 1

