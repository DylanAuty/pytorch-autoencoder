# utils.py
# Some utilities for the network training framework.

import torch
import torch.optim as optim

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