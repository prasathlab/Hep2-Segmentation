import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

results_dir_path = r"Z:\Balaji_Iyer\Projects\Hep-2_Segmentation\Results\GAN_Models"
res_dirs = [name for name in os.listdir(results_dir_path) if os.path.isdir(os.path.join(results_dir_path, name))]

all_1008_dir = "All_1008"
test_res_dir = "Test"

pattern_order = ['speckled ', 'nucleolar ', 'homogeneous ', 'golgi ', 'centromere ', 'numem ', 'mitsp ']

for res_dir in res_dirs:
	model_path = os.path.join(results_dir_path, res_dir)
	test_path = os.path.join(model_path, test_res_dir, "image_performances.tsv")
	test_save = os.path.join(model_path, test_res_dir, "Analysis")
	os.makedirs(test_save, exist_ok=True)
	test_df = pd.read_csv(test_path, sep="\t", index_col=0)

	fig, axes = plt.subplots()
	fig.set_size_inches(18, 9)
	sns.violinplot('pattern', 'Dice', data=test_df, ax=axes, scale='count', inner='point', order=pattern_order)
	axes.yaxis.grid(True)
	axes.set_ylim(bottom=0, top=1.1)
	axes.set_xlabel('Cell Staining Patterns')
	axes.set_ylabel('Dice Scores')
	plt.savefig(os.path.join(model_path, test_res_dir, "Analysis", "test_violin.png"), bbox_inches='tight', dpi=100)
	plt.close()

	fig, axes = plt.subplots()
	fig.set_size_inches(18, 9)
	ax = sns.boxplot(x="pattern", y="Dice", data=test_df,  order=pattern_order)
	ax = sns.swarmplot(x="pattern", y="Dice", data=test_df, color=".25",  order=pattern_order)
	axes.yaxis.grid(True)
	axes.set_ylim(bottom=0, top=1.1)
	axes.set_xlabel('Cell Staining Patterns')
	axes.set_ylabel('Dice Scores')
	plt.savefig(os.path.join(model_path, test_res_dir, "Analysis", "test_box.png"), bbox_inches='tight', dpi=100)
	plt.close()

	fig, axes = plt.subplots()
	fig.set_size_inches(18, 9)
	sns.displot(data=test_df, x="Dice", hue="pattern", kind='kde', hue_order=pattern_order)
	plt.savefig(os.path.join(model_path, test_res_dir, "Analysis", "test_kde.png"), bbox_inches='tight', dpi=100)
	plt.close()
	plt.close()

	fig, axes = plt.subplots()
	fig.set_size_inches(18, 9)
	sns.displot(data=test_df, x="Dice", hue="pattern", kind='hist', hue_order=pattern_order)
	plt.savefig(os.path.join(model_path, test_res_dir, "Analysis", "test_hist.png"), bbox_inches='tight', dpi=100)
	plt.close()
	plt.close()


	test_mean = test_df.groupby(by="pattern").mean()
	test_mean.to_csv(os.path.join(model_path, test_res_dir, "Analysis", "test_mean.tsv"), sep="\t")

	test_median = test_df.groupby(by="pattern").median()
	test_median.to_csv(os.path.join(model_path, test_res_dir, "Analysis", "test_median.tsv"), sep="\t")

	test_min = test_df.groupby(by="pattern").min()
	test_min.to_csv(os.path.join(model_path, test_res_dir, "Analysis", "test_min.tsv"), sep="\t")

	test_max = test_df.groupby(by="pattern").max()
	test_max.to_csv(os.path.join(model_path, test_res_dir, "Analysis", "test_max.tsv"), sep="\t")

	test_std = test_df.groupby(by="pattern").std()
	test_std.to_csv(os.path.join(model_path, test_res_dir, "Analysis", "test_std.tsv"), sep="\t")


	all_1008_path = os.path.join(model_path, all_1008_dir, "image_performances.tsv")
	all_1008_save = os.path.join(model_path, all_1008_dir, "Analysis")
	os.makedirs(all_1008_save, exist_ok=True)

	all_1008_df = pd.read_csv(all_1008_path, sep="\t", index_col=0)

	all_1008_mean = all_1008_df.groupby(by="pattern").mean()
	all_1008_mean.to_csv(os.path.join(model_path, all_1008_dir, "Analysis", "all_1008_mean.tsv"), sep="\t")

	all_1008_median = all_1008_df.groupby(by="pattern").median()
	all_1008_median.to_csv(os.path.join(model_path, all_1008_dir, "Analysis", "all_1008_median.tsv"), sep="\t")

	all_1008_min = all_1008_df.groupby(by="pattern").min()
	all_1008_min.to_csv(os.path.join(model_path, all_1008_dir, "Analysis", "all_1008_min.tsv"), sep="\t")

	all_1008_max = all_1008_df.groupby(by="pattern").max()
	all_1008_max.to_csv(os.path.join(model_path, all_1008_dir, "Analysis", "all_1008_max.tsv"), sep="\t")

	all_1008_std = all_1008_df.groupby(by="pattern").std()
	all_1008_std.to_csv(os.path.join(model_path, all_1008_dir, "Analysis", "all_1008_std.tsv"), sep="\t")

	fig, axes = plt.subplots()
	fig.set_size_inches(18, 9)
	sns.violinplot('pattern', 'Dice', data=all_1008_df, ax=axes, scale='count', inner='point', order=pattern_order)
	axes.yaxis.grid(True)
	axes.set_ylim(bottom=0)
	axes.set_xlabel('Cell Staining Patterns')
	axes.set_ylabel('Dice Scores')
	plt.savefig(os.path.join(model_path, all_1008_dir, "Analysis", "all_1008_violin.png"), bbox_inches='tight', dpi=100)
	plt.close()

	fig, axes = plt.subplots()
	fig.set_size_inches(18, 9)
	ax = sns.boxplot(x="pattern", y="Dice", data=all_1008_df, order=pattern_order)
	ax = sns.swarmplot(x="pattern", y="Dice", data=all_1008_df, color=".25", order=pattern_order)
	axes.yaxis.grid(True)
	axes.set_ylim(bottom=0)
	axes.set_xlabel('Cell Staining Patterns')
	axes.set_ylabel('Dice Scores')
	plt.savefig(os.path.join(model_path, all_1008_dir, "Analysis", "all_1008_box.png"), bbox_inches='tight', dpi=100)
	plt.close()

	fig, axes = plt.subplots()
	fig.set_size_inches(18, 9)
	sns.displot(data=all_1008_df, x="Dice", hue="pattern", kind='kde', hue_order=pattern_order)
	plt.savefig(os.path.join(model_path, all_1008_dir, "Analysis", "all_1008_kde.png"), bbox_inches='tight', dpi=100)
	plt.close()
	plt.close()

	fig, axes = plt.subplots()
	fig.set_size_inches(18, 9)
	sns.displot(data=all_1008_df, x="Dice", hue="pattern", kind='hist', hue_order=pattern_order)
	plt.savefig(os.path.join(model_path, all_1008_dir, "Analysis", "all_1008_hist.png"), bbox_inches='tight', dpi=100)
	plt.close()
	plt.close()
	debug = 1

