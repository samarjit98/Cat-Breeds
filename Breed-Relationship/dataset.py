from torch.utils.data import Dataset
import numpy as np 
import torch
from skimage import io
import os
import pickle

class CatsDataset(Dataset):
    def __init__(self, load=False):
        super(CatsDataset, self).__init__()
        
        self.catx = []
        self.caty = []

        if(load):
            cats = [
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
            src = '/home/nevronas/dataset/Pet_cats'

            i = 0

            for cat in cats:
                files = os.listdir(src + '/' + cat)
                for f in files:
                    if '.mat' in f:
                        continue
                    print(f)
                    image = io.imread(src + '/' + cat + '/' + f).astype('float')
                    image = np.array(image).astype(np.float32)
                    self.catx.append(image)
                    self.caty.append(i)
                i = i+1
            
            file_handler = open('./pickle/cats_pickled.dat', 'wb+')
            pickle.dump((self.catx, self.caty), file_handler)
            file_handler.close()

        else:
            file_handler = open('./pickle/cats_pickled.dat', 'rb+')
            self.catx, self.caty = pickle.load(file_handler)
            file_handler.close()


    def __len__(self):
        return len(self.catx)

    def __getitem__(self, idx):
        return torch.from_numpy(self.catx[idx]).float(), torch.Tensor(self.caty[idx])

if __name__ == '__main__':
    cats = CatsDataset(True)
