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
l_list = list(l_list[0].children())
features = torch.nn.Sequential(*l_list[:-1]) 

cats = [
		'cat1',
		'cat2'
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
		input = input.unsqueeze(0)
		input = torch.tensor(input).type(torch.FloatTensor)	#converting to tensor 
		input = input.to(device)
		images = features(input)
		label = 1
		for image in images:
		   image = image[0].detach().cpu().numpy()
		   image = image.transpose(1, 2, 0)
           image = image.transpose(1, 2, 0)
           cv2.imwrite("./{}/visuals_{}.png".format(cat_breeds[idx], label),image)
           label = label + 1            



if __name__=="__main__":
	visualization()