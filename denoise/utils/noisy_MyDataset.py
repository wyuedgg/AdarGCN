import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset

import random
import os
from PIL import Image
import numpy as np


def get_allPaths(data_dir, noisy_dir, clean_num):
    class_ids = os.listdir(os.path.join(data_dir, 'clean'))
    all_paths = {'clean': {}, 'noisy': {}}
    for class_id in class_ids:
        all_paths['clean'][class_id] = [os.path.join(data_dir, 'clean', class_id, p) for p in os.listdir(os.path.join(data_dir, 'clean', class_id))][:clean_num]
        all_paths['noisy'][class_id] = [os.path.join(data_dir, noisy_dir, class_id, p) for p in os.listdir(os.path.join(data_dir, noisy_dir, class_id))] 
       
    all_paths['map'] = {class_id: idx for idx, class_id in enumerate(class_ids)}
    all_paths['class_ids'] = class_ids
    all_paths['other'] = {class_id:[] for class_id in class_ids}
    for class_id in class_ids:
        for other_id in class_ids:
            if other_id != class_id:
                all_paths['other'][class_id] += all_paths['noisy'][other_id]

    return all_paths


class NoisyDataset(Dataset):

    def __init__(self, all_paths, class_id, transform, args):
        self.transform = transform 
        self.all_paths = all_paths
        self.cleanSize = args.cleanSize 
        self.dirtySize = args.dirtySize 
        self.noisySize = args.noisySize

        self.image_roots = []
        np.random.shuffle(self.all_paths['noisy'][class_id])
        np.random.shuffle(self.all_paths['other'][class_id])

        for batch_idx in range(int(len(self.all_paths['noisy'][class_id]) / self.noisySize)):
            np.random.shuffle(self.all_paths['clean'][class_id])
            self.image_roots += self.all_paths['clean'][class_id][:self.cleanSize]
            self.image_roots += self.all_paths['other'][class_id][batch_idx*self.dirtySize: (batch_idx+1)*self.dirtySize]
            self.image_roots += self.all_paths['noisy'][class_id][batch_idx*self.noisySize: (batch_idx+1)*self.noisySize]      

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        return image, image_root


def get_transform(is_training):

    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
    if is_training == False:
        return transforms.Compose([transforms.Resize((84,84)),
                                   transforms.ToTensor(),
                                   normalize])
    return transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                               transforms.RandomVerticalFlip(0.5),
                               transforms.RandomResizedCrop(84, scale=(0.8, 1.0)),
                               transforms.RandomGrayscale(0.5),
                               transforms.ToTensor(),
                               normalize])

def get_noisy_dataloader(all_paths, class_id, args, is_training=True):

    transform = get_transform(is_training)
    dataset = NoisyDataset(all_paths, class_id, transform, args)
    dataloader = DataLoader(dataset, batch_size=(args.cleanSize+args.dirtySize+args.noisySize), shuffle=False)

    return dataloader
