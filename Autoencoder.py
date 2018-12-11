# Autoencoder.py
# A basic autoencoder model that accepts 3 channel inputs and hopefully returns similarly dimensioned things at the other end.

import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):

	def __init__(self):
		super(Autoencoder, self).__init__()

		# self.encoder = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
		# for child in self.encoder.children():
		# 	for param in child.parameters():
		# 		param.requires_grad = False

		self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(8)
		self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(16)
		self.conv3 = nn.Conv2d(16,32, 3, padding=1)
		self.bn3 = nn.BatchNorm2d(32)
		self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
		self.bn4 = nn.BatchNorm2d(64)
		self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
		self.bn5 = nn.BatchNorm2d(128)
		self.conv6 = nn.Conv2d(128, 256, 3, padding=1)
		self.bn6 = nn.BatchNorm2d(256)
		self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
		self.bn7 = nn.BatchNorm2d(512)
		self.conv8 = nn.Conv2d(512, 1024, 3, padding=1)
		self.bn8 = nn.BatchNorm2d(1024)
		self.conv9 = nn.Conv2d(1024, 2048, 3, padding=1)
		self.bn9 = nn.BatchNorm2d(2048)


		self.deconv1 = nn.ConvTranspose2d(2048, 512, 3, padding=1, bias=False)
		self.dbn1 = nn.BatchNorm2d(512)
		self.deconv2 = nn.ConvTranspose2d(512, 256, 3, padding=1, bias=False)
		self.dbn2 = nn.BatchNorm2d(256)
		self.deconv3 = nn.ConvTranspose2d(256, 128, 3, padding=1, bias=False)
		self.dbn3 = nn.BatchNorm2d(128)
		self.deconv4 = nn.ConvTranspose2d(128, 64, 3, padding=1, bias=False)
		self.dbn4 = nn.BatchNorm2d(64)
		self.deconv5 = nn.ConvTranspose2d(64, 32, 3, padding=1, bias=False)
		self.dbn5 = nn.BatchNorm2d(32)
		self.deconv6 = nn.ConvTranspose2d(32, 16, 3, padding=1, bias=False)
		self.dbn6 = nn.BatchNorm2d(16)
		self.deconv7 = nn.ConvTranspose2d(16, 8, 3, padding=1, bias=False)
		self.dbn7 = nn.BatchNorm2d(8)
		self.deconv8 = nn.ConvTranspose2d(8, 3, 3, padding=1, bias=False)


	def forward(self, x):
		# print("FORWARD")
		# print(x.shape)
		# x = self.encoder(x)
		x = self.conv1(F.leaky_relu(x))
		x = self.bn1(x)
		# print(x.shape)
		x = self.conv2(x)
		x = self.bn2(F.leaky_relu(x))
		# print(x.shape)
		x = self.conv3(x)
		x = self.bn3(F.leaky_relu(x))
		# print(x.shape)
		x = self.conv4(x)
		x = self.bn4(F.leaky_relu(x))
		# print(x.shape)
		x = self.conv5(x)
		x = self.bn5(F.leaky_relu(x))
		# print(x.shape)
		x = self.conv6(x)
		x = self.bn6(F.leaky_relu(x))
		# print(x.shape)
		x = self.conv7(x)
		x = self.bn7(F.leaky_relu(x))
		# print(x.shape)
		x = self.conv8(x)
		x = self.bn8(F.leaky_relu(x))
		# print(x.shape)
		x = self.conv9(x)
		x = self.bn9(F.leaky_relu(x))
		# print(x.shape)		

		x = self.deconv1(F.relu(x))
		# print(x.shape)
		x = self.deconv2(F.relu(x))
		# print(x.shape)
		x = self.deconv3(F.relu(x))
		# print(x.shape)
		x = self.deconv4(F.relu(x))
		# print(x.shape)
		x = self.deconv5(F.relu(x))
		# print(x.shape)
		x = self.deconv6(F.relu(x))
		# print(x.shape)
		x = self.deconv7(F.relu(x))
		# print(x.shape)
		x = self.deconv8(F.relu(x))
		# print(x.shape)
		return x