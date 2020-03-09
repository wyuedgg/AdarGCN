from __future__ import print_function
from torchtools import *
import torch.utils.data as data
import random
import os
import numpy as np
from PIL import Image as pil_image
import pickle
from itertools import islice
from torchvision import transforms
import json
import scipy.io as sio


class MiniImagenetLoader(data.Dataset):
    def __init__(self, root, partition='train'):
        super(MiniImagenetLoader, self).__init__()
        # set dataset information
        self.root = root
        self.partition = partition
        
        self.data_size = [3, 84, 84]


        normalize = transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                            np.array([0.229, 0.224, 0.225]))
        
        # set transformer
        self.transform = transforms.Compose([transforms.Resize(92),
                                            transforms.RandomCrop(84),
                                            transforms.ToTensor(),
                                            normalize])

        # load data
        self.data = self.load_dataset()
        print('Dataset loaded')
        

    def trans_fname_img(self, path, data_size):

        img = pil_image.open(path)
        img = img.convert('RGB')

        return img

    
    def load_dataset(self): 
        """
        return a dict saving the information of csv
        :return: {label:[file1, file2 ...]}
        """
        partition = self.partition
        self.filename = {}
        #  general mode : full image

        class2imgs_dict = {}


        IMAGE_PATH = os.path.join(self.root, 'miniImagenet/images')
        csv_path = os.path.join(self.root, 'miniImagenet/split', partition + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        for l in lines:
            name, wnid = l.split(',')
            i_path = os.path.join(IMAGE_PATH, name)
            image_resized = self.trans_fname_img(i_path, self.data_size)

            if wnid not in class2imgs_dict.keys():
                class2imgs_dict[wnid] = []
                self.filename[wnid] = []
            class2imgs_dict[wnid].append(image_resized)
            self.filename[wnid].append(i_path)


        return class2imgs_dict


    def get_task_batch(self,
                       num_tasks=5,
                       num_ways=20,
                       num_shots=1,
                       num_queries=7,
                       seed=None):

        if seed is not None:
            random.seed(seed)

        # init task batch data
        support_data, support_label, query_data, query_label = [], [], [], []
        for _ in range(num_ways * num_shots):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            support_data.append(data)
            support_label.append(label)
        for _ in range(num_ways * num_queries):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            query_data.append(data)
            query_label.append(label)

        # get full class list in dataset
        full_class_list = list(self.data.keys())
        filenames = []
        # for each task
        for t_idx in range(num_tasks):
            # define task by sampling classes (num_ways)
            task_class_list = random.sample(full_class_list, num_ways)

            # for each sampled class in task
            for c_idx in range(num_ways):
                # sample data for support and query (num_shots + num_queries)
                #class_data_list = random.sample(self.data[task_class_list[c_idx]], num_shots + num_queries)
                class_data_list = random.sample(list(zip(self.filename[task_class_list[c_idx]] ,self.data[task_class_list[c_idx]])), num_shots + num_queries)


                # load sample for support set
                for i_idx in range(num_shots):
                    # set data
                    support_data[i_idx + c_idx * num_shots][t_idx] = self.transform(class_data_list[i_idx][1])
                    support_label[i_idx + c_idx * num_shots][t_idx] = c_idx

                # load sample for query set
                for i_idx in range(num_queries):
                    query_data[i_idx + c_idx * num_queries][t_idx] = self.transform(class_data_list[num_shots + i_idx][1])
                    filenames.append(class_data_list[num_shots + i_idx][0])
                    
                    query_label[i_idx + c_idx * num_queries][t_idx] = c_idx

        # <num_tasks, num_ways * (num_supports/num_queries), 3, 84, 84>
        # or <num_tasks, num_ways * (num_supports/num_queries)>
        support_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in support_data], 1)
        support_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in support_label], 1)
        query_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in query_data], 1)
        query_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in query_label], 1)

        return [support_data, support_label, query_data, query_label]#, filenames]



class CUBLoader(data.Dataset):
    def __init__(self, root, partition='train'):
        super(CUBLoader, self).__init__()
        # set dataset information
        self.root = root
        self.partition = partition
        
        self.data_size = [3, 84, 84]

        # set normalizer
        mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

         # set transformer
        self.transform = transforms.Compose([transforms.Resize(92),
                                            transforms.RandomCrop(84),
                                            transforms.ToTensor(),
                                            normalize])

        # load data

        self.data = self.load_dataset()

    def trans_fname_img(self, path, data_size):

        img = pil_image.open(path)
        img = img.convert('RGB')
        return img

    def load_dataset(self): 
        """
        return a dict saving the information of csv
        :return: {label:[file1, file2 ...]}
        """
        partition = self.partition
        
        
        self.filename = {}
        class_path = os.path.join(self.root, 'CUB_200_2011/images')
        class_dirs = os.listdir(class_path)
        class_dirs = sorted(class_dirs, key=lambda cls : int(cls.split('.')[0]))

        
        tr_val_te_split = sio.loadmat(os.path.join(self.root, 'CUB_200_2011/cub_split.mat'))
        train_cid, val_cid, test_cid = tr_val_te_split['train_cid'][0], tr_val_te_split['val_cid'][0], tr_val_te_split['test_cid'][0]
        part_cid = train_cid if partition == 'train' else (val_cid if partition == 'val' else test_cid)

        class2imgs_dict = {}
            
        for cid in part_cid:         # ./train/mode(train/test)/class_name(label)
            subdir = class_dirs[cid]
            imgs = os.listdir(os.path.join(class_path, subdir))

            for img in imgs:
                i_path = os.path.join(class_path, subdir, img)
                image_resized = self.trans_fname_img(i_path, self.data_size)

                if subdir not in class2imgs_dict.keys():
                    self.filename[subdir] = []
                    class2imgs_dict[subdir] = []
                self.filename[subdir].append(i_path)
                class2imgs_dict[subdir].append(image_resized)

        
        return class2imgs_dict



    def get_task_batch(self,
                       num_tasks=5,
                       num_ways=20,
                       num_shots=1,
                       num_queries=7,
                       seed=None):

        if seed is not None:
            random.seed(seed)

        # init task batch data
        support_data, support_label, query_data, query_label = [], [], [], []
        for _ in range(num_ways * num_shots):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            support_data.append(data)
            support_label.append(label)
        for _ in range(num_ways * num_queries):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            query_data.append(data)
            query_label.append(label)

        # get full class list in dataset
        full_class_list = list(self.data.keys())

        filenames = []
        # for each task
        for t_idx in range(num_tasks):

            # define task by sampling classes (num_ways)
            task_class_list = random.sample(full_class_list, num_ways)

            # for each sampled class in task
            for c_idx in range(num_ways):
                # sample data for support and query (num_shots + num_queries)
                class_data_list = random.sample(list(zip(self.filename[task_class_list[c_idx]] ,self.data[task_class_list[c_idx]])), num_shots + num_queries)


                # load sample for support set
                for i_idx in range(num_shots):
                    # set data
                    support_data[i_idx + c_idx * num_shots][t_idx] = self.transform(class_data_list[i_idx][1])
                    support_label[i_idx + c_idx * num_shots][t_idx] = c_idx

                # load sample for query set
                for i_idx in range(num_queries):
                    query_data[i_idx + c_idx * num_queries][t_idx] = self.transform(class_data_list[num_shots + i_idx][1])
                    filenames.append(class_data_list[num_shots + i_idx][0])
                    query_label[i_idx + c_idx * num_queries][t_idx] = c_idx

        # <num_tasks, num_ways * (num_supports/num_queries), 3, 84, 84>
        # or <num_tasks, num_ways * (num_supports/num_queries)>
        support_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in support_data], 1)
        support_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in support_label], 1)
        query_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in query_data], 1)
        query_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in query_label], 1)

        return [support_data, support_label, query_data, query_label]#, filenames]

