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
import argparse

# Imports from this folder
from Autoencoder import Autoencoder
import utils

# This may come in handy
# https://github.com/mortezamg63/Accessing-and-modifying-different-layers-of-a-pretrained-model-in-pytorch

def imshow(img):
	img = img / 2 + 0.5
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

def train(model, optimizer, criterion, trainset, batch_size=8, shuffle=True, epoch=0, num_epochs=2, checkpoint_dir="./checkpoints", save_freq=5):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

	if epoch != 0:
		print("Resuming training from epoch %d of %d." % (epoch, num_epochs))
		print("Batch size is %d, saving checkpoint to %s every %d epochs" % (batch_size, checkpoint_dir, save_freq))

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
				print('[E: %d, B: %2d] loss: %.3f, took %.3f secs' % (epoch + 1, i + 1, running_loss / 10, duration))
				running_loss = 0.0
				start_time = time.time()

		if epoch % save_freq == (save_freq - 1):
			print('Saving checkpoint for epoch %d' % (epoch + 1))
			# torch.save(model.state_dict(), './CIFAR10_checkpt_%d.pt' % epoch + 1)
			utils.saveCheckpoint(epoch, model, optimizer, loss, './CIFAR10_checkpt_%d.pt' % (epoch + 1))

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
	# Set up argparser.
	parser = argparse.ArgumentParser()
	parseTrainEval = parser.add_mutually_exclusive_group()
	parseTrainEval.add_argument("-t", "--train",														help="Use training mode",	action="store_true")
	parseTrainEval.add_argument("-e", "--evaluate",														help="Use evaluation mode", action="store_true")
	parser.add_argument(		"-b", "--batch_size",			type=int,	default=8,					help="Batch size to use for training or evaluation depending on what mode you're in")
	parser.add_argument(		"-s", "--save_frequency",		type=int,	default=5,					help="Save a checkpoint every SAVE_FREQUENCY epochs")
	parser.add_argument(		"-c", "--checkpoint_directory",	type=str,	default="./checkpoints", 	help="Directory to save checkpoints to")
	parser.add_argument(		"-n", "--num_epochs",			type=int,	default=50,					help="Number of epochs to train for")
	parser.add_argument(		"-l", "--load_checkpoint",		type=str,								help="Path of model checkpoint to load and use")

	args = parser.parse_args()

	print("Initialising...")
	# Transforms to put into a tensor and normalise the incoming Pillow images.
	transform = transforms.Compose(
		[	
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		]
	)

	print("	Loading datasets...")
	# Set up datasets, model and loss/optimiser. If there's cuda available then send to the GPU.
	trainset = torchvision.datasets.CIFAR10(root='~/WorkingDatasets', train=True, download=False, transform=transform)
	testset = torchvision.datasets.CIFAR10(root='~/WorkingDatasets', train=False, download=False, transform=transform)

	print("	Initialising model...")
	model = Autoencoder()
	epoch = 0	# This is used when resuming training and is overwritten on load.

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("	Using device:	" + str(device))
	
	if torch.cuda.device_count() > 1:
		print("	Using		%d CUDA-capable devices" % torch.cuda.device_count())
		model = nn.DataParallel(model)

	model.to(device)

	print("	Configuring optimiser...")
	criterion = nn.MSELoss()
	criterion = criterion.to(device)
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

	# To resume from a checkpoint:

	if(args.evaluate):
		print("\n### Evaluation Mode ###\n")
		if(args.load_checkpoint):
			print("Loading model checkpoint for evaluation from " + args.load_checkpoint)
			model, epoch, optimizer, loss = utils.loadCheckpoint(args.load_checkpoint, model)
		print("Evaluating model with batch size %d..." % args.batch_size)
		print(evaluate(model, criterion, testset, batch_size=args.batch_size))

	if(args.train):
		print("\n### Training Mode ###\n")
		if(args.load_checkpoint):
			print("Training from checkpoint: " + args.load_checkpoint)
		train(model, optimizer, criterion, trainset, batch_size=args.batch_size, epoch=epoch, num_epochs=args.num_epochs, save_freq=args.save_frequency, checkpoint_dir=args.checkpoint_directory)
	
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