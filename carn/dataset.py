import os
import glob
import h5py
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

def random_crop(hr, lr, size, scale):
    h, w = lr.shape[:-1]
    x = random.randint(0, w-size)
    y = random.randint(0, h-size)

    hsize = size*scale
    hx, hy = x*scale, y*scale

    crop_lr = lr[y:y+size, x:x+size].copy()
    crop_hr = hr[hy:hy+hsize, hx:hx+hsize].copy()
    #crop_test=crop_hr[::2,::2]
    #print(np.max(np.abs(crop_lr-crop_test)))

    return crop_hr, crop_lr


def random_flip_and_rotate(im1, im2):
    if random.random() < 0.5:
        im1 = np.flipud(im1)
        im2 = np.flipud(im2)

    if random.random() < 0.5:
        im1 = np.fliplr(im1)
        im2 = np.fliplr(im2)

    angle = random.choice([0, 1, 2, 3])
    im1 = np.rot90(im1, angle)
    im2 = np.rot90(im2, angle)

    # have to copy before be called by transform function
    return im1.copy(), im2.copy()


class TrainDataset(data.Dataset):
    def __init__(self, path, size, scale,fix_length=0,aug=1):
        super(TrainDataset, self).__init__()
        self.aug=aug
        self.size = size
        self.length=fix_length
        #print("01")
        h5f = h5py.File(path, "r")
        #print("02")
        #print(len(h5f["HR"].values()))
        self.hr = [v[:] for v in h5f["HR"].values()]
        self.hrlen=len(self.hr)
        #print("03")
        # perform multi-scale training
        if scale == 0:
            self.scale = [2, 3, 4]
            self.lr = [[v[:] for v in h5f["X{}".format(i)].values()] for i in self.scale]
        else:
            self.scale = [scale]
            self.lr = [[v[:] for v in h5f["X{}".format(scale)].values()]]
       # print("04")
        h5f.close()
        #print(len(self.hr))
        #print(len(self.lr[0]))

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
       

    def __getitem__(self, index):
        if self.length:
            index= index % self.hrlen
        size = self.size
        
        item = [(self.hr[index], self.lr[i][index]) for i, _ in enumerate(self.lr)]
       
        item = [random_crop(hr, lr, size, self.scale[i]) for i, (hr, lr) in enumerate(item)]
        if self.aug:
            item = [random_flip_and_rotate(hr, lr) for hr, lr in item]
        

        
        return [(self.transform(hr), self.transform(lr)) for hr, lr in item]

    def __len__(self):
        if self.length:
            return self.length
        else:
            return self.hrlen
        

class TestDataset(data.Dataset):
    def __init__(self, dirname, scale):
        super(TestDataset, self).__init__()

        self.name  = dirname.split("/")[-1]
        self.scale = scale
        if "DIV" in self.name:
            self.hr = glob.glob(os.path.join("{}_valid_HR".format(dirname), "*.png"))
            self.lr = glob.glob(os.path.join("{}_valid_LR_bicubic".format(dirname), 
                                             "X{}/*.png".format(scale)))
        else:
            '''
            all_files = glob.glob(os.path.join(dirname, "x{}/*.png".format(scale)))
            self.hr = [name for name in all_files if "HR" in name]
            self.lr = [name for name in all_files if "LR" in name]
            '''
            self.hr = glob.glob(os.path.join(dirname, "hr_test","*.dat"))
            self.lr = glob.glob(os.path.join(dirname, "lr_test","*.dat"))
            
            

        self.hr.sort()
        self.lr.sort()
        #print(len(self.hr))
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        if "DIV" in self.name:
            hr = Image.open(self.hr[index])
            lr = Image.open(self.lr[index])
        else:
            hr=np.fromfile(self.hr[index],dtype=np.float32).reshape((1800,3600,1))
            lr=np.fromfile(self.lr[index],dtype=np.float32).reshape((900,1800,1))
        
        filename = self.hr[index].split("/")[-1]

        return self.transform(hr), self.transform(lr), filename

    def __len__(self):
        return len(self.hr)
