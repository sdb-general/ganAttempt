#https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from IPython.display import HTML
# from inspect import currentframe, getframeinfo

mypath = './torchtest.pth'

# plt.figure(figsize = (8,8))
# plt.imshow(train[0])
# plt.show()

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

batch_size = 50
image_size = 28
nc = 1 #number of channels, these are single channel images
nz = 100 #size of latent z vector
ngf = 64 #size of feature maps in the generator, assuming must be same as image size (edit: this was wrong)
ndf = 64 #same but discrim #change back to 28 if using first versin of discrim
num_epochs = 25
lr = 0.0002 #learnig rate
lrg = 0.001
d_iters = 5
beta1 = 0.5 #some para i don't understand
ngpu = 1 #cpu mode

mnist_trainset = dset.MNIST(root='./data', train=True, download=True, transform=None)
train_data = mnist_trainset.data
dataset = torch.utils.data.TensorDataset(train_data/256, mnist_trainset.train_labels)
dataloader = torch.utils.data.DataLoader(dataset,
										batch_size=batch_size,
										shuffle=True) #workers is number of threads for the dataloader, unsure of function



#########################################

#weight initialise
#takes an initialised model and renormalises all layers

#########################################


def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02) #mean = 0, stdev = 0.02
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)

#########################################

#generator defn

#########################################


class Generator(nn.Module):
	def __init__(self, ngpu):
		super(Generator, self).__init__()
		self.label_emb = nn.Embedding(10, 10) #second arg could be something else
		self.main = nn.Sequential(
			nn.ConvTranspose2d(nz + 10, ngf * 8, 4, 1, 0, bias = False), #(input channels, outoput chanenls, size of kernel, stride length, padding = cutoff at end of otput...)
			nn.BatchNorm2d(ngf * 8),
			nn.ReLU(True),
			#######################################################
			nn.ConvTranspose2d(ngf * 8, ngf*4, 4, 2, 1, bias=False), 
			nn.BatchNorm2d(ngf*4),
			nn.ReLU(True),
			#######################################################
			nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 2),
			nn.ReLU(True),
			#######################################################
			# nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
			# nn.BatchNorm2d(ngf),
			# nn.ReLU(True),
			#######################################################
			# nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
			# nn.Tanh()
			nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 3, bias=False),
			nn.Sigmoid()
			)
	def forward(self, inp, labels):
		lab = self.label_emb(labels).unsqueeze(-1).unsqueeze(-1)
		# print(inp.shape, lab.shape)
		x = torch.cat([inp, lab], 1)
		return self.main(x)



class Discriminator(nn.Module):
	def __init__(self,ngpu):
		super(Discriminator, self).__init__()
		self.label_emb = nn.Embedding(10, 10)
		self.ngpu = ngpu
		self.main = nn.Sequential(
			nn.Conv2d(nc + 10, ndf * 2, 4, 2, 3, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			########################################
			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			########################################
			nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
			########################################
			nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
			nn.Sigmoid()
		)
	def forward(self, inp, labels):
		lab = self.label_emb(labels).unsqueeze(-1).unsqueeze(-1)
		lab = lab.repeat(1,1,28,28)
		# print(inp.shape, lab.shape)
		x = torch.cat([inp, lab], 1)
		return self.main(x)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
netG = Generator(ngpu).to(device)
netG.apply(weights_init)


# y = netG(torch.randn(1,nz, 1,1), torch.randint(10,(1,)))


# print(netD)
# y = torch.randn(1, 1, 28, 28)
# x = netD(y, torch.randint(10,(1,)))
# # print(x.squeeze())
# raise ValueError


#########################################

#initalise BCE Loss function

#########################################

criterion = nn.MSELoss()

#create batch of latent vectors

fixed_noise = torch.randn(50, nz, 1, 1, device=device)

#conventions

real_label = 1.
fake_label = 0. 

#setup adam optimisers for G and D

optimiserD = optim.Adam(netD.parameters(), lr=lr, betas= (beta1, 0.999))
optimiserG = optim.Adam(netG.parameters(), lr=lrg, betas= (beta1, 0.999))

########################################

#time to train

########################################

# begin training loop

#list to track progress

img_list = []
G_losses = []
D_losses = []
iters = 0

print(device)
def train():
	print('starting the training loop . ... ... . .')
	for epoch in range(num_epochs):
		print('\n beginning epoch number ', epoch + 1)
		for index, (data, num_label) in enumerate(dataloader, 0):
			######################################
			#update D network, maximing its loss
			#training with real batch
			######################################
			#print(data.shape, label.shape)
			# raise ValueError

			netD.zero_grad()
			real_cpu = data.to(device).unsqueeze(1).float()
			num_label = num_label.to(device)
			#print(real_cpu.shape)
			b_size = real_cpu.size(0)
			label = torch.full((b_size,), 
								real_label,
								dtype=torch.float,
								device=device)
			#forward pass real batch through D
			output = netD(real_cpu, num_label).view(-1)
			#calculate loss
			errD_real = criterion(output, label)
			errD_real.backward()
			D_x = output.mean().item()

			####################################
			#training with fake batch
			####################################

			#generate batch of latent vectors
			noise = torch.randn(batch_size, nz, 1, 1, device = device)
			#generate fake image batch with G
			fake = netG(noise, num_label)
			label.fill_(fake_label)
			#classify with D
			output = netD(fake.detach(), num_label).view(-1)
			#calc D loss on the fake batch

			errD_fake = criterion(output, label)
			#gradients
			errD_fake.backward()
			D_G_Z1 = output.mean().item()
			#add gradients from reals and fakes
			errD = errD_fake + errD_real

			##################################
			#update D
			##################################
			optimiserD.step()

			##################################
			#update G network
			##################################
			if index % d_iters == 0:

				netG.zero_grad()
				label.fill_(real_label) #fake labels are real for generator cost

				#we just updated D, so we do another forward pass of the fake batch through D
				output = netD(fake, num_label).view(-1)
				#calc G's loss
				errG = criterion(output, label)
				#calc gradients
				errG.backward()
				D_G_Z2 = output.mean().item()
				#update G
				optimiserG.step()

			#output training stats
			if index % 50 == 0:
				print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
	                  % (epoch, num_epochs, index, len(dataloader),
	                     errD.item(), errG.item(), D_x, D_G_Z1, D_G_Z2))
				x = netG(torch.randn(1, nz, 1, 1).to(device), torch.randint(9, (1,)).to(device))
				plt.figure()
				plt.imshow(x.detach().cpu().numpy().squeeze(0).squeeze(0))
				plt.savefig('epochNumber{}'.format(epoch + 1))
				plt.close()
			G_losses.append(errG.item())
			D_losses.append(errD.item())

			

			#check how the gen is doing by saving outputs on fixed_noise

			#skipped this part
	# torch.save(netD.state_dict(), mypath)
	# torch.save(netG.state_dict(), mypath)



train()

# mynet = Generator(ngpu).to(device)
# mynet.load_state_dict(torch.load(mypath))
# print(torch.randint(9,(1,)))
# x = mynet(torch.randn(1, nz, 1, 1),torch.randint(9,(1,)))

# plt.imshow(x.detach().numpy().squeeze(0).squeeze(0))
# plt.show()


