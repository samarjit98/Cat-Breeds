from torch.utils.data import Dataset
import numpy as np 
import torch

class IndicDataset(Dataset):
    def __init__(self, X_train, y_train):
        super(IndicDataset, self).__init__()
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_train = np.expand_dims(X_train,axis = 1)
        self.X_tens = torch.from_numpy(self.X_train)
        self.y_tens = torch.from_numpy(self.y_train)
        self.count = 0

    def __len__(self):
        return int(self.y_train.shape[0])

    def __getitem__(self, idx):
        return self.X_tens[idx].type(torch.FloatTensor), self.y_tens[idx].type(torch.LongTensor)
