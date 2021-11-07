import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.nn import init
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import time
import os
from PIL import *
import cv2
import glob
import pdb

class Hep2TrainValDataset(Dataset):
	"""Hep2 Dataset for Training and Validation"""
	def __init__(self, img_dir, mask_dir, transform=None):
		'''
		if you want to use augmentations, the easiest way is to instantiate a Dataset class with aug_df.
		In the train loop, for each step in an epoch. Train 2 times, once with orig dataset and once with aug dataset
		'''
		"""
			Args:
				tsv_file (string): Path to the csv file containing image and mask details.
				transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.img_dir = img_dir
		self.mask_dir = mask_dir
		self.transform = transform
		#Create a dict with {"ImageName": img_name, "MaskName" mask_name}
		self.img_files = glob.glob(self.img_dir + "/*.npy")
		self.img_mask_df = pd.DataFrame(columns=["imageNames", "maskNames"])
		for idx, img_filepath in enumerate(self.img_files):
			img_basename = os.path.basename(img_filepath)
			img_name = os.path.splitext(img_basename)[0]
			mask_name = img_name + "_mask.npy"
			mask_filepath = os.path.join(self.mask_dir, mask_name)
			self.img_mask_df.loc[idx, "imageNames"] = img_filepath
			self.img_mask_df.loc[idx, "maskNames"] = mask_filepath


	def __len__(self):
		return self.img_mask_df.shape[0]

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		img_path = self.img_mask_df.loc[idx, "imageNames"]
		mask_path = self.img_mask_df.loc[idx, "maskNames"]
		img = np.load(img_path)
		mask = np.load(mask_path)
		sample = {'image': img, 'mask': mask}
		img = np.einsum('ijk -> kij', img)
		mask = np.einsum('ijk -> kij', mask)
		if self.transform:
			sample = self.transform(sample)

		return img, mask















