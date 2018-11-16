import torch
import torch as tt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# This may come in handy
# https://github.com/mortezamg63/Accessing-and-modifying-different-layers-of-a-pretrained-model-in-pytorch

def imshow(img):
	img = img / 2 + 0.5
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

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

def saveCheckpoint(epoch, model, optimizer, loss, path):
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

# def loadCheckpoint(path):
# 	model = Autoencoder()
# 	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 	checkpoint = torch.load(path)
# 	model.load_state_dict

def main():
	print ("Hello there")

	def transform(inTensor):
		transforml2 = transforms.Compose(
			[	
				transforms.ToTensor(),
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			]
		)
		x = transforml2(inTensor)
		# x = x.repeat(3, 1, 1)
		return x

	trainset = torchvision.datasets.CIFAR10(root='~/WorkingDatasets', train=True, download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=40, shuffle=True, num_workers=2)
	testset = torchvision.datasets.CIFAR10(root='~/WorkingDatasets', train=False, download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)

	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	net = Autoencoder()

	if torch.cuda.device_count() > 1:
		net = nn.DataParallel(net)

	# net.to(device)

	# criterion = nn.MSELoss()
	# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	# for epoch in range(40):
	# 	running_loss = 0.0
	# 	for i, data in enumerate(trainloader, 0):
	# 		inputs, labels = data
	# 		inputs = inputs.to(device)

	# 		optimizer.zero_grad()

	# 		outputs = net(inputs)
	# 		loss = criterion(outputs, inputs)
	# 		loss.backward()
	# 		optimizer.step()

	# 		running_loss += loss.item()
	# 		if i % 10 == 9:
	# 			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
	# 			running_loss = 0.0

	# 	if epoch % 5 == 4:
	# 		print('Saving checkpoint for epoch %d' % (epoch + 1))
	# 		# torch.save(net.state_dict(), './CIFAR10_checkpt_%d.pt' % epoch + 1)
	# 		saveCheckpoint(epoch, net, optimizer, loss, './CIFAR10_checkpt_%d.pt' % (epoch + 1))


	# torch.save(net.state_dict(), './CIFAR10_trained_40.pt')

	dataiter = iter(testloader)
	images, labels = dataiter.next()
	images = images.to(device)

	loadedNet = Autoencoder()
	loadedNet = nn.DataParallel(loadedNet)
	state = loadedNet.state_dict()
	state.update(torch.load('./CIFAR10_checkpt_40.pt').state_dict)

	loadedNet.load_state_dict(state)
	loadedNet.to(device)

	with torch.no_grad():
		trainedOutput = loadedNet(images)

	concatSlice = torch.cat((images, trainedOutput.detach())).cpu()

	imshow(torchvision.utils.make_grid(concatSlice))


if __name__ == "__main__":
	main()