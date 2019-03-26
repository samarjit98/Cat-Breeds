from torch.utils.data import Dataset
import numpy as np 
import torch
import cv2
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
            src = '/home/nevronas/dataset/Pet_cats2'

            desired_size = 300
            '''
            for cat in cats:
                files = os.listdir(src + '/' + cat)
                for f in files:
                    try:
                        if '.mat' in f:
                            continue
                        print(f)
                        image = cv2.imread(src + '/' + cat + '/' + f)
                        curr_max = max(image.shape[:2])
                        desired_size = max(desired_size, curr_max)
                    except:
                        continue
            '''
            i = 0

            for cat in cats:
                files = os.listdir(src + '/' + cat)
                for f in files:
                    try:
                        if '.mat' in f:
                            continue
                        print(f)
                        image = cv2.imread(src + '/' + cat + '/' + f)
                        old_size = image.shape[:2]
                        ratio = float(desired_size)/max(old_size)
                        new_size = tuple([int(x*ratio) for x in old_size])
                        image = cv2.resize(image, (new_size[1], new_size[0]))
                        delta_w = desired_size - new_size[1]
                        delta_h = desired_size - new_size[0]
                        top, bottom = delta_h//2, delta_h-(delta_h//2)
                        left, right = delta_w//2, delta_w-(delta_w//2)
                        color = [0, 0, 0]
                        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
                        image = image.transpose(1, 2, 0)
                        image = image.transpose(1, 2, 0)
                        print(image.shape)
                        self.catx.append(image)
                        self.caty.append(i)
                    except:
                        continue
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
        one_hot = torch.zeros(12)
        one_hot = one_hot.scatter(0, torch.tensor(self.caty[idx]).type(torch.LongTensor), 1)
        return torch.tensor(self.catx[idx]).type(torch.FloatTensor), torch.tensor(one_hot).type(torch.LongTensor)

if __name__ == '__main__':
    cats = CatsDataset(True)
