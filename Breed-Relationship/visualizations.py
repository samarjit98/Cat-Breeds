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
