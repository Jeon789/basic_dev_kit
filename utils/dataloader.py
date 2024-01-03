import os, time
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, RandomCrop



class SIDD_RN_DS(Dataset):
    def __init__(self, config, files, transforms=None, mode=None):
        super().__init__()
        self.config = config
        self.files = files
        self.transforms = transforms
        self.mode = mode
    
    def __getitem__(self, index):
        real_noise = self.files[index]

        if self.transforms:
            real_noise = self.transforms(real_noise)

        return real_noise

    def __len__(self):
        return len(self.files)
    
        # inputs = []
        # targets = []
        # for _ in range(self.config['data_per_oneday']):
        #     sample = self.data[index]
        #     sample = self.cut_window(sample)
        #     sample.rename(columns=COL2NAME_SIONYU, inplace=True)
        #     input, target = sample[self.input_cols], sample[self.target_col]
        #     input, target = torch.tensor(input.values), torch.tensor(target.values)
        #     input, target = torch.transpose(input, 0,1), torch.transpose(target, 0,1)  # channel(cols) x length(86400)
        #     if self.transforms:
        #         input_transform, target_transform = self.transforms
        #         input = input_transform(input)
        #         target = target_transform(target)
        #     inputs.append(input.type(torch.float32))
        #     targets.append(target.mean(axis=-1).type(torch.float32))
        # inputs = torch.stack(inputs)
        # targets = torch.stack(targets)
        # return inputs, targets
    
    
    


class SIDD_RN_DL(DataLoader):
    def __init__(self, config, dataset, mode=None, *args, **kwargs):
    # def __init__(self, dataset, config, *args, cols=None, seconds=None, **kwargs):
        self.mode = mode
        if self.mode == 'test':
            self.batch_size = config['test_batch_size']
            self.shuffle = False
        else:
            self.batch_size = config['train_batch_size']
            self.shuffle = True


        super(SIDD_RN_DL,self).__init__(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=config['num_workers'], drop_last=config['drop_last'], *args, **kwargs)
        self.seconds = config['seconds']



def build_loader(config):

    print('Start building loader')
    start = time.time()

    all_file = get_all_file(config['data_dir'])
    train_files, val_files, test_files = split(all_file, config['split_ratio'])

    transform = transforms.Compose([
        OpenPNG(),
        ToTensor(),
        RandomCrop(config['image_size']),
        Gaussian_Perturbation(config['sigma'])
        ])

    train, val, test = SIDD_RN_DS(config, train_files, transforms=transform), SIDD_RN_DS(config, val_files, transforms=transform), SIDD_RN_DS(config, test_files, transforms=transform)
    train_loader, val_loader, test_loader = SIDD_RN_DL(config, train, mode='train'), SIDD_RN_DL(config, val, mode='test'), SIDD_RN_DL(config, test, mode='test')

    print(f'It takes {time.time()-start:.2f} seconds for dataset&dataloader')

    return train_loader, val_loader, test_loader


def get_all_file(DATA_ROOT):
    return [os.path.join(DATA_ROOT, file) for file in os.listdir(DATA_ROOT)]

def split(all_file, split_ratio):
    split_ratio = np.array(split_ratio)
    split_ratio = split_ratio / sum(split_ratio)
    a, b, c = split_ratio
    L = len(all_file)
    a, b, c = int(a*L), int(b*L), int(c*L)
    return all_file[:a], all_file[a:a+b], all_file[a+b:]

###################################################################################################
# transforms    

class OpenPNG(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return Image.open(sample)
    


class Gaussian_Perturbation(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, sample):
        if self.sigma == 'adaptive' :
            return sample + torch.std(sample)*torch.randn(sample.size())
        else:
            return sample + self.sigma*torch.randn(sample.size())
    