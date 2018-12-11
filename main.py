import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import time

from Autoencoder import Autoencoder

# This may come in handy
# https://github.com/mortezamg63/Accessing-and-modifying-different-layers-of-a-pretrained-model-in-pytorch

def imshow(img):
	img = img / 2 + 0.5
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

def saveCheckpoint(model, epoch, optimizer, loss, path):
	torch.save(
		{
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': loss,
		},
		path
	)
	print('Checkpoint saved to ' + path)
	
def loadCheckpoint(path, model):
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

	checkpoint = torch.load(path)
	state = model.state_dict()
	state.update(checkpoint['model_state_dict'])
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']

	return model, epoch, optimizer, loss

def train(model, optimizer, criterion, trainset, batch_size=8, shuffle=True, epoch=0, num_epochs=2):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

	if epoch != 0:
		print("Resuming training from epoch %d of %d." % (epoch, num_epochs))

	for epoch in range(epoch, num_epochs):
		running_loss = 0.0
		start_time = time.time()
		for i, data in enumerate(trainloader, 0):
			inputs, labels = data

			inputs = inputs.to(device)

			optimizer.zero_grad()

			outputs = model(inputs)
			loss = criterion(outputs, inputs)
			loss = loss.to(device)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			if i % 10 == 9:
				duration = time.time() - start_time
				print('[%d, %5d] loss: %.3f, took %.3f secs' % (epoch + 1, i + 1, running_loss / 10, duration))
				running_loss = 0.0
				start_time = time.time()

		if epoch % 5 == 4:
			print('Saving checkpoint for epoch %d' % (epoch + 1))
			# torch.save(model.state_dict(), './CIFAR10_checkpt_%d.pt' % epoch + 1)
			saveCheckpoint(epoch, model, optimizer, loss, './CIFAR10_checkpt_2_%d.pt' % (epoch + 1))

def evaluate(model, criterion, testset, batch_size=8):
	print("Evaluating model performance")
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# if torch.cuda.device_count() > 1:
	# 	model = nn.DataParallel(model)

	# model = model.to(device)
	model.eval()

	# Set up the different loss functions we need here:
	# 	RMSE Lin,
	# 	RMSE log,
	#	abs rel,
	#	square rel,
	# 	and %age of points s.t. max((y/y*), (y*/y)) < delta for delta = 1.25, 1.25^2, and 1.25^3.

	# Each of these is a tensor, and will have its elements summed at the end for efficiency.
	batchSumOfSquareDiffs = 0
	batchSumOfSquareDiffsOfLogs = 0
	batchSumOfAbsRelDiffs = 0
	batchSumOfSquareRelDiffs = 0

	# These are tallies for elements that satisfy the delta inequalities - they will be tensors until the end, when they'll have their elements summed to get the outputs.
	totalDelta1_25 = 0
	totalDelta1_25_2 = 0
	totalDelta1_25_3 = 0

	# These are used to keep track of the number of items we have in the test set, necessary for metrics.
	totalBatches = 0
	totalSamples = 0
	totalPoints = 0

	eps = 1e-3	# This is a tiny value that will be used as the minimum value for thing that will be log'd

	with torch.no_grad():
		for i, data in enumerate(testloader, 0):
			inputs, labels = data			

			# Send inputs to device, push through network. Clamp both inputs and outputs to avoid both /0 errors and log(x<=0) errors.
			# The transforms applied to the data on the way in can sometimes make 0 values in the inputs into very small -ve values.
			inputs = torch.clamp(inputs.to(device), min=eps)
			outputs = torch.clamp(model(inputs), min=eps)

			# outputs = inputs	# TESTING LINE

			currBatchSquareDiffs = torch.pow(outputs - inputs, 2)
			batchSumOfSquareDiffs += currBatchSquareDiffs
			batchSumOfSquareDiffsOfLogs += torch.pow(torch.log(outputs) - torch.log(inputs), 2)
			batchSumOfAbsRelDiffs += torch.abs(outputs - inputs)
			batchSumOfSquareRelDiffs += (currBatchSquareDiffs / outputs)
			# print(currBatchSquareDiffs/outputs)

			totalDelta1_25 += torch.lt(torch.max(outputs/inputs, inputs/outputs), 1.25)
			totalDelta1_25_2 += torch.lt(torch.max(outputs/inputs, inputs/outputs), 1.5625)
			totalDelta1_25_3 += torch.lt(torch.max(outputs/inputs, inputs/outputs), 1.953125)

			totalBatches += 1

		totalSamples = totalBatches * batch_size
		totalPoints = torch.numel(totalDelta1_25) * totalBatches

		# Calculate our normalised %age (0 to 1) of points satisfying the various deltas in the above inequality.
		delta1_25 = torch.sum(totalDelta1_25).item() / totalPoints
		delta1_25_2 = torch.sum(totalDelta1_25_2).item() / totalPoints	
		delta1_25_3 = torch.sum(totalDelta1_25_3).item() / totalPoints

		# Calculate the rest of the metrics mentioned above
		RMSELin = torch.sqrt(torch.sum(batchSumOfSquareDiffs) / totalPoints)
		RMSELog = torch.sqrt(torch.sum(batchSumOfSquareDiffsOfLogs) / totalPoints)
		AbsRel = torch.sum(batchSumOfAbsRelDiffs) / totalPoints
		SquareRel = torch.sum(batchSumOfSquareRelDiffs) / totalPoints

		return(RMSELin.item(), RMSELog.item(), AbsRel.item(), SquareRel.item(), delta1_25, delta1_25_2, delta1_25_3)

def main():
	# Transforms to put into a tensor and normalise the incoming Pillow images.
	transform = transforms.Compose(
		[	
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		]
	)

	# Set up datasets, model and loss/optimiser. If there's cuda available then send to the GPU.
	trainset = torchvision.datasets.CIFAR10(root='~/WorkingDatasets', train=True, download=False, transform=transform)
	testset = torchvision.datasets.CIFAR10(root='~/WorkingDatasets', train=False, download=False, transform=transform)

	model = Autoencoder()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)

	model.to(device)

	criterion = nn.MSELoss()
	criterion = criterion.to(device)
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

	# To resume training:
	model, epoch, optimizer, loss = loadCheckpoint('checkpoints/CIFAR10_checkpt_40.pt', model)

	print(evaluate(model, criterion, testset, batch_size=80))

	# train(model, optimizer, criterion, trainset, batch_size=40, epoch=epoch, num_epochs=100)
	
	# Load a model, shove a batch of 8 through and display the results
	# testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)

	# dataiter = iter(testloader)
	# images, labels = dataiter.next()
	# images = images.to(device)

	# net = Autoencoder()
	# net = nn.DataParallel(net)

	# net, epoch, optimizer, loss = loadCheckpoint('./checkpoints/CIFAR10_checkpt_40.pt', net)
	# print("Loading at epoch %d" % epoch)
	# # state = net.state_dict()
	# # state.update(torch.load('./CIFAR10_checkpt_40.pt').state_dict)

	# # net.load_state_dict(state)
	# net.to(device)

	# with torch.no_grad():
	# 	trainedOutput = net(images)

	# concatSlice = torch.cat((images, trainedOutput.detach())).cpu()

	# imshow(torchvision.utils.make_grid(concatSlice))


if __name__ == "__main__":
	main()