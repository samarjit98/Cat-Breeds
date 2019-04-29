import os
import gc
import torch
import argparse
import librosa
import matplotlib
import numpy as np
from collections import Counter
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import save_image


from models import *
from dataset import *
from utils import progress_bar
from time import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('==> Loading network..')
rowcnn = RowCNN().to(device)
rowcnn.load_state_dict(torch.load('./weights/network.ckpt'))
l_list = list(rowcnn.children())
print(l_list)
#features = torch.nn.Sequential(*l_list[:-1])
features = l_list[0]
print(features) 

cats = [
		'../../../../../dataset/Pet_cats2/Abyssinian/Abyssinian_80.jpg',
		'../../../../../dataset/Pet_cats2/Bengal/Bengal_147.jpg',
		'../../../../../dataset/Pet_cats2/Birman/Birman_86.jpg',
		'../../../../../dataset/Pet_cats2/Bombay/Bombay_22.jpg',
		'../../../../../dataset/Pet_cats2/British_Shorthair/British_Shorthair_18.jpg',
		'../../../../../dataset/Pet_cats2/Egyptian_Mau/Egyptian_Mau_183.jpg',
		'../../../../../dataset/Pet_cats2/Maine_Coon/Maine_Coon_164.jpg',
		'../../../../../dataset/Pet_cats2/Persian/Persian_94.jpg',
		'../../../../../dataset/Pet_cats2/Ragdoll/Ragdoll_10.jpg',
		'../../../../../dataset/Pet_cats2/Russian_Blue/Russian_Blue_119.jpg',
		'../../../../../dataset/Pet_cats2/Siamese/Siamese_146.jpg',
		'../../../../../dataset/Pet_cats2/Sphynx/Sphynx_233.jpg'
	]

cat_breeds = [
        'Abyssinian',
        'Bengal',
        'Birman',
        'Bombay',
        'British_Shorthair',
        'Egyptian_Mau',
        'Maine_Coon',
        'Persian',
        'Ragdoll',
        'Russian_Blue',
        'Siamese',
        'Sphynx'
    ]

def visualization():

	for catbreed in cat_breeds:
		os.makedirs("./{}".format(catbreed), exist_ok=True)

	for idx in range(len(cats)):			
		input = cv2.imread(cats[idx])
		input = input.transpose(1, 2, 0) 
		input = input.transpose(1, 2, 0) 
		input = torch.tensor(input).type(torch.FloatTensor)	#converting to tensor 
		input = input.unsqueeze(0)
		input = input.to(device)
		print(input.shape)
		images = []
		for conv in features:
			images.append(conv(input))
		label = 1
		for image in images:
			image = image[0].detach().cpu().numpy()
			for i in range(image.shape[0]):
				imagecp = image[i]
				#image = image.transpose(1, 2, 0)
				#image = image.transpose(1, 2, 0)
				print(imagecp.shape)
				cv2.imwrite("./{}/visuals_{}.png".format(cat_breeds[idx], label),imagecp)
				label = label + 1            



if __name__=="__main__":
	visualization()
