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

import matplotlib.pyplot as plt
import matplotlib

os.makedirs('./images', exist_ok=True)
os.makedirs('./checkpoint', exist_ok=True)
os.makedirs('./logs', exist_ok=True)
os.makedirs('./weights', exist_ok=True)
os.makedirs('./information', exist_ok=True)

parser = argparse.ArgumentParser(description='PyTorch Cat Breed Relation')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate') 
parser.add_argument('--batch_size', default=8, type=int) 
parser.add_argument('--epochs', '-e', type=int, default=15, help='Number of epochs to train.')
parser.add_argument('--preparedata', type=int, default=1)

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('==> Preparing data..')

criterion = nn.CrossEntropyLoss()

print('==> Creating networks..')
alexnet = AlexNet(12).to(device)

print('==> Loading data..')
trainset = CatsDataset()

def train_breeds(currepoch, epoch):
    dataloader2 = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    dataloader = iter(dataloader2)
    print('\n=> Breed Epoch: %d' % epoch)
    
    train_loss, correct, total = 0, 0, 0
    params = alexnet.parameters()
    optimizer = optim.Adam(params, lr=args.lr)

    for batch_idx in range(len(dataloader)):
        inputs, targets = next(dataloader)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        y_pred = alexnet(inputs)
        loss = criterion(y_pred, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = y_pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        with open("./logs/breed_train_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(train_loss / total))

        with open("./logs/breed_train_acc.log", "a+") as afile:
            afile.write("{}\n".format(correct / total))

        del inputs
        del targets
        gc.collect()
        torch.cuda.empty_cache()
        torch.save(alexnet.state_dict(), './weights/network.ckpt')
        with open("./information/info.txt", "w+") as f:
            f.write("{} {}".format(epoch, batch_idx))
        progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    torch.save(alexnet.state_dict(), './checkpoints/network_epoch_{}.ckpt'.format(epoch + 1))
    print('=> Classifier Network : Epoch [{}/{}], Loss:{:.4f}'.format(currepoch, epoch, train_loss / len(dataloader)))

print('==> Training starts..')
for epoch in range(args.epochs):
    train_breeds(epoch, args.epochs)
   
