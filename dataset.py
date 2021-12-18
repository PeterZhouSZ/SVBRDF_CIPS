__all__ = ['MultiScaleDataset',
           'ImageDataset'
           ]

from io import BytesIO
import math

import lmdb
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
import torch.nn.functional as F

import tensor_transforms as tt
from torchvision import transforms    
import random

class MultiScaleDataset(Dataset):
    def __init__(self, path, integer_values=False, pat_c=1, pat_index=-1, emb_pat='blur',resize=False):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.integer_values = integer_values
        self.emb_pat = emb_pat
        self.pat_c = pat_c if 'net' not in emb_pat else 1
        self.pat_index = pat_index
        self.resize_train = resize
        # for i in range(self.pat_c-1):
        #     setattr(self, f'Gblurrer_{i}', transforms.GaussianBlur(kernel_size=(5, 5), sigma=0.5))

        # self.n = resolution // crop_size
        # self.log_size = int(math.log(self.n, 2))
        # self.to_crop = True if crop_size
        # self.coords = tt.convert_to_coord_format(1, resolution, resolution, integer_values=self.integer_values)

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = {}

        with self.env.begin(write=False) as txn:
            key = f'{str(index).zfill(7)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.totensor(img).unsqueeze(0)

        # ----------------- prcess svbrdf inputs ----------------- 
        
        b,c,h,w = img.shape
        # print('h', h)
        # print('w', w)
        if w == 5*h:
            img = self.normalize(img)
            img = torch.cat((img[:,:,:,h:2*h],img[:,:,:,2*h:3*h],img[:,0:1,:,3*h:4*h],img[:,:,:,4*h:5*h]), dim=1)

        else:
            # data augmentation
            color_jitter = transforms.Compose([transforms.ToPILImage(), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.5, hue=0.5), transforms.ToTensor()])

            D = []
            for i in range(img.shape[0]):
                D .append(color_jitter(img[i,:,:,h:2*h]))

            D = torch.stack(D, dim=0)
            gamma = 0.8+random.random()*0.4 # [0.8~1.2]

            H = img[:,0:1,:,0:h]**gamma
            R = img[:,0:1,:,2*h:3*h]**gamma


            # load patterns based on index or not
            if self.pat_index>=0:
                i = self.pat_index+3
                P_list = img[:,0:1,:,i*h:(i+1)*h]

            else:
                if self.emb_pat=='ff_blur':
                    P_list = []
                    for i in range(self.pat_c):
                        i = i+3
                        P = img[:,0:1,:,i*h:(i+1)*h]
                        P_list.append(P)

                    P_list = torch.cat(P_list, dim=1)

                else:
                    P_list = []
                    for i in range(self.pat_c):
                        i = i+3
                        P = img[:,0:1,:,i*h:(i+1)*h]
                        P_list.append(P)

                    P_list = torch.cat(P_list, dim=1)
                # print('...................... pattern', P_list)

            img = torch.cat((P_list, H, D, R), dim=1) # [1,C,H, W]
            img = img*2-1
            print('load data', img.shape)

            if self.resize_train:
                img = F.interpolate(img, size=(256,256), mode='bilinear', align_corners=True)

            return img.squeeze(0) #, self.coords.squeeze(0)


class ImageDataset(Dataset):
    def __init__(self, path, transform, resolution=256, to_crop=False):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.crop = tt.RandomCrop(resolution)
        self.to_crop = to_crop

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{str(index).zfill(7)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)
        if self.to_crop:
            img = self.crop(img.unsqueeze(0)).squeeze(0)

        return img
