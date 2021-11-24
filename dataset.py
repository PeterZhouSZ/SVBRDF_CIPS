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

import tensor_transforms as tt
from torchvision import transforms    
import random

class MultiScaleDataset(Dataset):
    def __init__(self, path, crop_size=64, integer_values=False):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.crop_size = crop_size
        self.integer_values = integer_values
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

    def _crop(self, x):

        b,c,h,w = x.shape

        w0 = random.randint(-self.crop_size,w)
        h0 = random.randint(-self.crop_size,w)

        wc = w0 + self.crop_size
        hc = h0 + self.crop_size

        p = torch.zeros((b,c,self.crop_size,self.crop_size))
        # print('p shape', p.shape)
        # print(w, ' rand: ', w0, h0)

        # seperate crop and stitch them manually
        # [7 | 8 | 9]
        # [4 | 5 | 6]
        # [1 | 2 | 3]
        # 1
        if h0<=0 and w0<=0:
            p[:,:,0:-h0,0:-w0] = x[:,:, h+h0:h, w+w0:w]
            p[:,:,-h0:,0:-w0] = x[:,:, 0:hc, w+w0:w]
            p[:,:,0:-h0,-w0:] = x[:,:, h+h0:h, 0:wc]
            p[:,:,-h0:,-w0:] = x[:,:, 0:hc, 0:wc]
        # 2
        elif h0<=0 and (w0<w-self.crop_size and w0>0):
            p[:,:,0:-h0,:] = x[:,:, h+h0:h,w0:wc]
            p[:,:,-h0:,:] = x[:,:, 0:hc, w0:wc]
        # 3
        elif h0<=0 and w0 >=w-self.crop_size:
            p[:,:,0:-h0,0:w-w0] = x[:,:, h+h0:h, w0:w]
            p[:,:,-h0:,0:w-w0] = x[:,:, 0:hc, w0:w]
            p[:,:,0:-h0,w-w0:] = x[:,:, h+h0:h, 0:wc-w]
            p[:,:,-h0:,w-w0:] = x[:,:, 0:hc, 0:wc-w]

        # 4
        elif (h0>0 and h0<h-self.crop_size) and w0<=0:
            p[:,:,:,0:-w0] = x[:,:, h0:hc, w+w0:w]
            p[:,:,:,-w0:] = x[:,:, h0:hc, 0:wc]
        # 5
        elif (h0>0 and h0<h-self.crop_size) and (w0<w-self.crop_size and w0>0):
            p = x[:,:, h0:hc, w0:wc]
        # 6
        elif (h0>0 and h0<h-self.crop_size) and w0 >=w-self.crop_size:
            p[:,:,:,0:w-w0] = x[:,:, h0:hc, w0:w]
            p[:,:,:,w-w0:] = x[:,:, h0:hc, 0:wc-w]

        # 7
        elif h0 >=h-self.crop_size and w0<=0:
            p[:,:,0:h-h0,0:-w0] = x[:,:, h0:h, w+w0:w]
            p[:,:,h-h0:,0:-w0] = x[:,:, 0:hc-h, w+w0:w]
            p[:,:,0:h-h0,-w0:] = x[:,:, h0:h, 0:wc]
            p[:,:,h-h0:,-w0:] = x[:,:, 0:hc-h, 0:wc]
        # 8
        elif h0 >=h-self.crop_size and (w0<w-self.crop_size and w0>0):
            p[:,:,0:h-h0,:] = x[:,:, h0:h,w0:wc]
            p[:,:,h-h0:,:] = x[:,:, 0:hc-h, w0:wc]
        # 9
        elif h0 >=h-self.crop_size and w0 >=w-self.crop_size:
            p[:,:,0:h-h0,0:w-w0] = x[:,:, h0:h, w0:w]
            p[:,:,h-h0:,0:w-w0] = x[:,:, 0:hc-h, w0:w]
            p[:,:,0:h-h0,w-w0:] = x[:,:, h0:h, 0:wc-w]
            p[:,:,h-h0:,w-w0:] = x[:,:, 0:hc-h, 0:wc-w]

        del x

        return p


    def __getitem__(self, index):
        data = {}

        with self.env.begin(write=False) as txn:
            key = f'{str(index).zfill(7)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.totensor(img).unsqueeze(0)

        # if self.to_crop:
        #     img = self.crop_resolution(img)

        # ----------------- prcess svbrdf inputs ----------------- 
        
        b,c,h,w = img.shape
        # print('h', h)
        # print('w', w)
        if w == 5*h:
            img = self.normalize(img)
            img = torch.cat((img[:,:,:,h:2*h],img[:,:,:,2*h:3*h],img[:,0:1,:,3*h:4*h],img[:,:,:,4*h:5*h]), dim=1)

        elif w==4*h:

            # data augmentation
            # print(img[:,:,:,2*h:3*h].shape)
            color_jitter = transforms.Compose([transforms.ToPILImage(), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.5, hue=0.5), transforms.ToTensor()])

            D = []
            for i in range(img.shape[0]):
                # D = torch.cat([D, color_jitter(img[i,:,:,2*h:3*h]).unsqueeze(0)], dim=0)
                D .append(color_jitter(img[i,:,:,2*h:3*h]))

            D = torch.stack(D, dim=0)
            # print(D.shape)
            gamma = 0.8+random.random()*0.4 # [0.8~1.2]

            H = img[:,0:1,:,h:2*h]**gamma
            R = img[:,0:1,:,3*h:4*h]**gamma

            img = torch.cat((H, D, R), dim=1)
            img = img*2-1

            # img = self.normalize(img)

        # h is greater than crop size, then crop, otherwise no
        if h > self.crop_size:
            img = self._crop(img)

        return img.squeeze(0)#, self.coords.squeeze(0)


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
