#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 13:24:56 2021

@author: mibook
"""
import glob
import os, pdb
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import elasticdeform
from torchvision import transforms

def fetch_loaders(processed_dir, batch_size=32, use_channels=[0,1],
                  train_folder='train', val_folder='val', test_folder='',
                  shuffle=True):
    """ Function to fetch dataLoaders for the Training / Validation
    Args:
        processed_dir(str): Directory with the processed data
        batch_size(int): The size of each batch during training. Defaults to 32.
    Return:
        Returns train and val dataloaders
    """
    normalize = False
    train_dataset = CoastalDataset(processed_dir / train_folder, use_channels, normalize,
                                    transforms = transforms.Compose([ 
                                    #    Grayscale(1),
                                    #    Brightness(0.5),
                                    #    Contrast(0.5),
                                    #    Standardize(1)
                                    #           Rot270(0.3),
                                    #           Cut(0.5)
                                           ])
                                    )
    val_dataset = CoastalDataset(processed_dir / val_folder, use_channels, normalize)
    #num_workers train:8, val:3, test:3
    loader = {
        "train": DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, shuffle=shuffle, pin_memory=True),
        "val": DataLoader(val_dataset, batch_size=batch_size,
                          num_workers=0, shuffle=False, pin_memory=True)}

    if test_folder:
        test_dataset = CoastalDataset(processed_dir / test_folder, use_channels)
        loader["test"] = DataLoader(test_dataset, batch_size=batch_size,
                                    num_workers=1, shuffle=shuffle)

    return loader

class CoastalDataset(Dataset):
    """Custom Dataset for Coastal Data
    Indexing the i^th element returns the underlying image and the associated
    binary mask
    """

    def __init__(self, folder_path, use_channels, normalize=False, transforms=None):
        """Initialize dataset.
        Args:
            folder_path(str): A path to data directory
        """

        self.img_files = glob.glob(os.path.join(folder_path, '*img*'))
        self.mask_files = [s.replace("img", "label") for s in self.img_files]
        self.use_channels = use_channels
        self.normalize = normalize
        self.transforms = transforms
        if self.normalize:
            arr = np.load(folder_path.parent / "normalize.npy")
            self.mean, self.std = arr[0][use_channels], arr[1][use_channels]

    def __getitem__(self, index):

        """ getitem method to retrieve a single instance of the dataset
        Args:
            index(int): Index identifier of the data instance
        Return:
            data(x) and corresponding label(y)
        """
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        data = np.load(img_path)
        data = data[:,:,self.use_channels]  
        if self.normalize:
            data = (data - self.mean) / self.std
        label = np.load(mask_path)
        zeros = label == 0
        ones = label == 1
        label = np.concatenate((zeros, ones), axis=2)
        sample = {'image': data, 'mask': label}
        if self.transforms:
            sample = self.transforms(sample)
        data = torch.from_numpy(sample['image'].copy()).float()
        label = torch.from_numpy(sample['mask'].copy()).float()
        return data, label

    def __len__(self):
        """ Function to return the length of the dataset
            Args:
                None
            Return:
                len(img_files)(int): The length of the dataset (img_files)
        """
        return len(self.img_files)


class FlipHorizontal(object):
    """Flip horizontal randomly the image in a sample.

    Args:
        p (float between 0 and 1): Probability of FlipHorizontal
    """
    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample['image'], sample['mask']
        if torch.rand(1) < self.p:
            data = data[:, ::-1, :]
            label = label[:, ::-1, :]
        return {'image': data, 'mask': label}

class FlipVertical(object):
    """Flip vertically randomly the image in a sample.

    Args:
        p (float between 0 and 1): Probability of FlipVertical
    """
    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample['image'], sample['mask']
        if torch.rand(1) < self.p:
            data = data[::-1, :, :]
            label = label[::-1, :, :]
        return {'image': data, 'mask': label}

class Rot270(object):
    """Flip vertically randomly the image in a sample.

    Args:
        p (float between 0 and 1): Probability of Rot270
    """
    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample['image'], sample['mask']
        if torch.rand(1) < self.p:
            data = data.transpose((1,0,2))
            label = label.transpose((1,0,2))
        return {'image': data, 'mask': label}

class Cut(object):
    """Cut randomly the first 7 channels of image in a sample.

    Args:
        p (float between 0 and 1): Probability of FlipHorizontal
    """
    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample['image'], sample['mask']
        channels = data.shape[2]
        if torch.rand(1) < self.p:
            prob = torch.rand(1)
            if prob <= 0.25:
                data[:256, :256, :channels] = 0
                label[:256, :256, :] = 0
            elif prob <= 0.5:
                data[:256, 256:, :channels] = 0
                label[:256, 256:, :] = 0
            elif prob <= 0.75:
                data[256:, :256, :channels] = 0
                label[256:, :256, :] = 0
            else:
                data[256:, 256:, :channels] = 0
                label[256:, 256:, :] = 0
        return {'image': data, 'mask': label}

class ElasticDeform(object):
    """Apply Elasticdeform from U-Net
    """
    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample['image'], sample['mask']
        if torch.rand(1) < self.p:
            [data, label] = elasticdeform.deform_random_grid([data, label], axis=(0, 1))
        return {'image': data, 'mask': label}

class Contrast(object):
    """Apply brightness distortion
    """
    
    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Value should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample['image'], sample['mask']
        start = np.random.uniform()*0.5
        end = np.random.uniform()*0.5 + 0.5
        data = np.clip(data, start, end)*self.p + (1-self.p)*data
        return {'image': data, 'mask': label}

class Brightness(object):
    """Apply brightness distortion
    """
    
    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample['image'], sample['mask']
        data = data + (self.p * (np.random.uniform() - 0.5))
        data = np.clip(data, 0, 1)
        return {'image': data, 'mask': label}

class Grayscale(object):
    """Apply brightness distortion
    """
    
    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample['image'], sample['mask']
        if torch.rand(1) < self.p:
            temp = np.mean(data, axis=-1)
            data = np.zeros(data.shape)
            for i in range(data.shape[2]): data[:,:,i] = temp
        return {'image': data, 'mask': label}

class Standardize(object):
    """Apply brightness distortion
    """
    
    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample['image'], sample['mask']
        if torch.rand(1) < self.p:
            data=data-np.min(data)
            data=data/np.max(data)
        return {'image': data, 'mask': label}

class DropoutChannels(object):
    """Apply Random channel dropouts
    """
    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample['image'], sample['mask']
        if torch.rand(1) < self.p:
            rand_channel_index = np.random.randint(low = 0, high = data.shape[2])
            data[:, :, rand_channel_index] = 0
        return {'image': data, 'mask': label}
