from torch.utils.data.dataset import Dataset

import os
import torch
import torch.nn.functional as F
import fnmatch
import numpy as np
import random

class NYUv2(Dataset):
    def __init__(self, root, do_semseg, do_depth, do_normal, train=True, augmentation=False):
        self.train = train
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation
        self.do_semseg = do_semseg
        self.do_depth = do_depth
        self.do_normal = do_normal

        # read the data file
        if train:
            self.data_path = root + '/train'
        else:
            self.data_path = root + '/val'

        # calculate data length
        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))

    def __getitem__(self, index):
        x = {}

        # load data from the pre-processed npy files
        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0))
        x['image'] = image
        if self.do_semseg:
            semantic = torch.from_numpy(np.load(self.data_path + '/label/{:d}.npy'.format(index)))
            x['semseg'] = semantic
        if self.do_depth:
            depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0))
            x['depth'] = depth
        if self.do_normal:
            normal = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/normal/{:d}.npy'.format(index)), -1, 0))
            x['normals'] = normal

        # apply data augmentation if required
        if self.augmentation:
            if torch.rand(1) < 0.5:
                x['image'] = torch.flip(x['image'], dims=[2])
                if self.do_semseg:
                    x['semseg'] = torch.flip(x['semseg'], dims=[1])
                if self.do_depth:
                    x['depth'] = torch.flip(x['depth'], dims=[2])
                if self.do_normal:
                    x['normals'] = torch.flip(x['normals'], dims=[2])
                    x['normals'][0, :, :] = - x['normals'][0, :, :]

        x = {k:v.float() for k,v in x.items()}
        return x

    def __len__(self):
        return self.data_len

class CityScapes(Dataset):
    def __init__(self, root, do_semseg, do_depth, train=True, augmentation=False):
        self.train = train
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation
        self.do_semseg = do_semseg
        self.do_depth = do_depth

        # read the data file
        if train:
            self.data_path = root + '/train'
        else:
            self.data_path = root + '/val'

        # calculate data length
        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))

    def __getitem__(self, index):
        x = {}

        # load data from the pre-processed npy files
        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0))
        x['image'] = image
        if self.do_semseg:
            semantic = torch.from_numpy(np.load(self.data_path + '/label_19/{:d}.npy'.format(index)))
            x['semseg'] = semantic
        if self.do_depth:
            depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0))
            x['depth'] = depth

        # apply data augmentation if required
        if self.augmentation:
            if torch.rand(1) < 0.5:
                x['image'] = torch.flip(x['image'], dims=[2])
                if self.do_semseg:
                    x['semseg'] = torch.flip(x['semseg'], dims=[1])
                if self.do_depth:
                    x['depth'] = torch.flip(x['depth'], dims=[2])

        x = {k:v.float() for k,v in x.items()}
        return x

    def __len__(self):
        return self.data_len