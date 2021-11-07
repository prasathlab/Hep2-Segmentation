import numpy as np
import torch
import torch.nn as nn



# According to pix-pix
## ACCORDING TO GANS PAPER
class Discriminator(nn.Module):
	def __init__(self):
		'''
		class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
							  dilation=1, groups=1, bias=True,
							   padding_mode='zeros', device=None, dtype=None
							   )
		'''
		super(Discriminator, self).__init__()
		self.conv1 = nn.Conv2d(1, 64, 4, stride=2, padding=1)
		self.relu1 = nn.LeakyReLU(0.2)

		self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
		self.bn1 = nn.BatchNorm2d(128)
		self.relu2 = nn.LeakyReLU(0.2)

		self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
		self.bn2 = nn.BatchNorm2d(256)
		self.relu3 = nn.LeakyReLU(0.2)

		self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)
		self.bn3 = nn.BatchNorm2d(512)
		self.relu4 = nn.LeakyReLU(0.2)

		self.conv5 = nn.Conv2d(512, 1, 4, 1, 1)
		# self.bn4 = nn.BatchNorm2d(self.out_channels)
		# self.relu5 = nn.LeakyReLU(0.2)
		self.fcClass = nn.Linear(225, 7)
		self.fcDis = nn.Linear(225, 1)

		self.logSoftmax = nn.LogSoftmax(dim=1)
		self.softmax = nn.Softmax(dim=1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = self.conv1(x)
		x = self.relu1(x)

		x = self.conv2(x)
		x = self.bn1(x)
		x = self.relu2(x)

		x = self.conv3(x)
		x = self.bn2(x)
		x = self.relu3(x)

		x = self.conv4(x)
		x = self.bn3(x)
		x = self.relu4(x)

		x = self.conv5(x)
		# x = self.bn4(x)
		# x = self.relu5(x)

		flat = x.view(-1, 225)
		#fcClass is 7 dimensional...Basically the class label of the generated image
		fcClass = self.fcClass(flat)
		#fcDis is 1 dim...Basically real or fake
		fcDis = self.fcDis(flat)

		classes = self.logSoftmax(fcClass)
		predic = self.softmax(fcClass)
		# check
		realorfake = self.sigmoid(fcDis).view(-1, 1).squeeze(1)

		# realorfake=self.sigmoid(fcDis)

		#return realorfake, classes, predic
		#For now we return only real or fake. Later we can add class labels
		return realorfake

	# weight initial problem
	def init_weights(m):
		if type(m) == nn.Conv2d:
			torch.nn.init.normal_(0.0, 0.02)

		elif type(m) == nn.BatchNorm2d:
			m.weight.data.normal_(1.0, 0.02)
			m.bias.data.fill_(0)




