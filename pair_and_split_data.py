import yaml
import glob
from file_utils import *
import psutil
import pdb

pdb.set_trace()
yaml_file = r"/data/aronow/Balaji_Iyer/Projects/Hep-2_Segmentation/src/configs/exp1_cluster.yaml"
with open(yaml_file, 'r') as stream:
    try:
        cfg = yaml.load(stream)
        print(cfg)
    except yaml.YAMLError as exc:
        print(exc)

# ---------------------------------------------------------------------------------------------
pid = os.getpid()
py = psutil.Process(pid)
memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
print('memory use:', memoryUse)
pdb.set_trace()

#Load the train images.
all_files = glob.glob(cfg["pair_and_split_data"]["data_dir"] + "/*.tif")
pdb.set_trace()
paired_df = pair_image_mask_names(all_files)
paired_df.to_csv("all_paired_names.tsv", sep="\t")

pdb.set_trace()
test_ratio = cfg["pair_and_split_data"]["test_ratio"]
n_test_samples = round(paired_df.shape[0]*test_ratio)
test_df = paired_df.sample(n=n_test_samples, replace=False)
train_val_df = paired_df[~paired_df.isin(test_df)].dropna()
pdb.set_trace()

val_ratio = cfg["pair_and_split_data"]["val_ratio"]
n_val_samples = round(paired_df.shape[0]*val_ratio)
val_df = train_val_df.sample(n=n_val_samples, replace=False)
train_df = train_val_df[~train_val_df.isin(val_df)].dropna()

test_df.to_csv("test_data.tsv", sep="\t")
val_df.to_csv("val_data.tsv", sep="\t")
train_df.to_csv("train_data.tsv", sep="\t")
pdb.set_trace()


memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
print('memory use:', memoryUse)
pdb.set_trace()

debug = 1

