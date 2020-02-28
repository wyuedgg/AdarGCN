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
    def __init__(self, root, partition='train', mode='general', k1_clean=50, k2_add=600, threshold=0.5):
        super(MiniImagenetLoader, self).__init__()
        # set dataset information
        self.root = root
        self.partition = partition
        self.mode = mode
        self.k1_clean = k1_clean
        self.k2_add = k2_add
        self.threshold = threshold
        self.data_size = [3, 84, 84]

        # set normalizer
        mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        # set transformer
        if self.partition == 'train':
            self.transform = transforms.Compose([transforms.RandomCrop(84, padding=4),
                                                 lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])
        else:  # 'val' or 'test' ,
            self.transform = transforms.Compose([lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])

        # load data
        self.data = self.load_dataset()
        

        if True:
            self.data_flat = [(k, item) for k, l in self.data.items() for item in l]
        print('---------- exp_mode: {} ---------- partition: {} ---------'.format(self.mode, self.partition))

    def trans_fname_img(self, path, data_size):

        img = pil_image.open(path)
        img = img.convert('RGB')
        img = img.resize((84, 84), pil_image.ANTIALIAS)
        img = np.array(img, dtype='float32')

        image2resize = pil_image.fromarray(np.uint8(img))
        image_resized = image2resize.resize((data_size[2], data_size[1]))
        return image_resized

    

    def load_dataset(self): 
        # mode in ['general', 'clean', 'addnoisy', 'subnoisy']
        # general mode: full images
        # clean:        only clean images, k1 images per class
        # addnoisy:    clean images plus noisy images = k1 + k2
        """
        return a dict saving the information of csv
        :return: {label:[file1, file2 ...]}
        """
        partition = self.partition
        if partition != 'train':
            self.mode = 'general'
        self.filename = {}
        #  general mode : full image
        if self.mode == 'general':
            class2imgs_dict = {}
            class_i = 0
            path = os.path.join(self.root, 'mini-imagenet', partition)  # ./root/mode(train/test)
            subdirs = os.listdir(path)
            for subdir in subdirs:         # ./train/mode(train/test)/class_name(label)
                imgs = os.listdir(os.path.join(path, subdir))

                for img in imgs:
                    i_path = os.path.join(path, subdir, img)
                    image_resized = self.trans_fname_img(i_path, self.data_size)
                    #image_data = np.array(image_data, dtype='float32')

                    #image_data = np.transpose(image_data, (2, 0, 1))

                    if class_i not in class2imgs_dict.keys():
                        class2imgs_dict[class_i] = []
                        self.filename[class_i] = []
                    class2imgs_dict[class_i].append(image_resized)
                    self.filename[class_i].append(i_path)

                class_i += 1

            return class2imgs_dict

        elif self.mode == 'clean':
            class2imgs_dict = {}
            class_i = 0
            path = os.path.join(self.root, 'mini-imagenet', partition)  # ./root/mode(train/test)
            subdirs = os.listdir(path)
            for subdir in subdirs:         # ./train/mode(train/test)/class_name(label)
                imgs = os.listdir(os.path.join(path, subdir))
                counter = 0
                for img in imgs:
                    i_path = os.path.join(path, subdir, img)
                    image_resized = self.trans_fname_img(i_path, self.data_size)
                    if class_i not in class2imgs_dict.keys():
                        class2imgs_dict[class_i] = []
                    class2imgs_dict[class_i].append(image_resized)

                    counter += 1
                    if counter >= self.k1_clean:
                        break

                #print(len(class2imgs_dict[class_i]))
                class_i += 1

            return class2imgs_dict

        elif self.mode == 'addnoisy':
            class2imgs_dict = {}
            class_i = 0
            path = os.path.join(self.root, 'mini-imagenet', partition)  # ./root/mode(train/test)
            subdirs = os.listdir(path)
            for subdir in subdirs:         # ./train/mode(train/test)/class_name(label)
                imgs = os.listdir(os.path.join(path, subdir))
                counter = 0
                for img in imgs:
                    i_path = os.path.join(path, subdir, img)
                    image_resized = self.trans_fname_img(i_path, self.data_size)
            
                    if class_i not in class2imgs_dict.keys():
                        class2imgs_dict[class_i] = []
                    class2imgs_dict[class_i].append(image_resized)

                    counter += 1
                    if counter >= self.k1_clean:
                        break

                t_path = os.path.join(self.root, 'mini-mix/noisy_'+str(self.k2_add), subdir)
                #print(t_path)
                imgs = os.listdir(t_path)
                imgs = sorted(imgs, key=lambda img : int(img.split('.')[0]))
                add_counter = 0
                for img in imgs:
                    i_path = os.path.join(t_path, img)
                    image_resized = self.trans_fname_img(i_path, self.data_size)
                    class2imgs_dict[class_i].append(image_resized)

                    add_counter += 1
                    if add_counter >= self.k2_add:
                        break

                #print(len(class2imgs_dict[class_i]))
                class_i += 1

            return class2imgs_dict

        elif self.mode == 'subnoisy':

            print('------K1 clean: {}, k2_add: {}, threshold: {}-----'.format(self.k1_clean, self.k2_add, self.threshold))
            denoisy_fname = os.path.join(self.root, 'mini_clean_50_noisy1200.json') # LP
           
            f = open(denoisy_fname, 'r')
            print(denoisy_fname)

            sub_noisy_imgfile = json.load(f)

            class2imgs_dict = {}
            class_i = 0
            path = os.path.join(self.root, 'mini-imagenet', partition)  # ./root/mode(train/test)
            subdirs = os.listdir(path)
            for subdir in subdirs:         # ./train/mode(train/test)/class_name(label)
                imgs = os.listdir(os.path.join(path, subdir))
                counter = 0
                for img in imgs:
                    i_path = os.path.join(path, subdir, img)
                    image_resized = self.trans_fname_img(i_path, self.data_size)
    
                    if class_i not in class2imgs_dict.keys():
                        class2imgs_dict[class_i] = []
                    class2imgs_dict[class_i].append(image_resized)

                    counter += 1
                    if counter >= self.k1_clean:
                        break
                t_path = os.path.join(self.root, 'mini-mix/noisy_'+str(self.k2_add), subdir)
                
                
                # according to threshold
                imgs = os.listdir(t_path)
                for img in imgs:

                    cof = sub_noisy_imgfile[subdir][img] 
                    #print(self.threshold)
                    if cof < self.threshold:
                        continue

                    i_path = os.path.join(t_path, img)
                    image_resized = self.trans_fname_img(i_path, self.data_size)
                    class2imgs_dict[class_i].append(image_resized)
                
                '''
                # according to portion
                choosed = 0
                scored_imgs = sub_noisy_imgfile[subdir]
                sorted_imgs = sorted(scored_imgs, key=scored_imgs.__getitem__, reverse=True) 
                for img in sorted_imgs:

                    i_path = os.path.join(t_path, img)
                    image_resized = self.trans_fname_img(i_path, self.data_size)
                    class2imgs_dict[class_i].append(image_resized)

                    #print(img, scored_imgs[img])
                    choosed += 1
                    if choosed >= int(self.k2_add*0.6):
                        break
                '''
                #print(len(class2imgs_dict[class_i]))
                class_i += 1

            f.close()
            return class2imgs_dict


    def __getitem__(self, index):
        label, img = self.data_flat[index]
        img = self.transform(img)
        #print(img.size())
        return img, int(label)

    def __len__(self):
        return len(self.data_flat)

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
    def __init__(self, root, partition='train', mode='general', k1_clean=20, k2_add=600, threshold=0.5, val_cid=None):
        super(CUBLoader, self).__init__()
        # set dataset information
        self.root = root
        self.partition = partition
        self.mode = mode
        self.k1_clean = k1_clean

        self.k2_add = k2_add
        self.threshold = threshold
        self.val_cid = val_cid
        self.data_size = [3, 84, 84]

        # set normalizer
        mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        # set transformer
        if self.partition == 'train':
            self.transform = transforms.Compose([transforms.RandomCrop(84, padding=4),
                                                 lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])
        else:  # 'val' or 'test' ,
            self.transform = transforms.Compose([lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])

        # load data

        self.data = self.load_dataset()
        print('---------- exp_mode: {} ---------- partition: {} ---------'.format(self.mode, self.partition))
        print(self.data.keys())

    def trans_fname_img(self, path, data_size):

        img = pil_image.open(path)
        img = img.convert('RGB')
        img = img.resize((84, 84), pil_image.ANTIALIAS)
        img = np.array(img, dtype='float32')

        image2resize = pil_image.fromarray(np.uint8(img))
        image_resized = image2resize.resize((data_size[2], data_size[1]))
        return image_resized

    def load_dataset(self): 
        # mode in ['general', 'addnoisy', 'subnoisy']
        # general mode: full images
        # addnoisy:    clean images plus noisy images = full + k2
        """
        return a dict saving the information of csv
        :return: {label:[file1, file2 ...]}
        """
        partition = self.partition
        if partition != 'train':
            self.mode = 'general'
        
        self.filename = {}
        class_path = os.path.join(self.root, 'CUB_200_2011/images')
        class_dirs = os.listdir(class_path)
        class_dirs = sorted(class_dirs, key=lambda cls : int(cls.split('.')[0]))

        # CUB train/val split randomly
        '''
        traintestplit = sio.loadmat(os.path.join(self.root, 'CUB_200_2011/train_test_split.mat')) # 1 - 200
        train_cid = (traintestplit['train_cid'][0] - 1).tolist()
        test_cid = (traintestplit['test_cid'][0] - 1).tolist()
        if self.val_cid is None:
            val_cid = random.sample(train_cid, 50)
        else:
            val_cid = self.val_cid
        train_cid = list(set(train_cid).difference(set(val_cid)))
        part_cid = train_cid if partition == 'train' else (val_cid if partition == 'val' else test_cid)
        '''
        tr_val_te_split = sio.loadmat(os.path.join(self.root, 'CUB_200_2011/cub_split.mat'))
        train_cid, val_cid, test_cid = tr_val_te_split['train_cid'][0], tr_val_te_split['val_cid'][0], tr_val_te_split['test_cid'][0]
        part_cid = train_cid if partition == 'train' else (val_cid if partition == 'val' else test_cid)

        
        class2imgs_dict = {}
        #  general mode : full image
        if self.mode == 'general':
            
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

        elif self.mode == 'clean':
            
            for cid in part_cid:         # ./train/mode(train/test)/class_name(label)
                subdir = class_dirs[cid]
                imgs = os.listdir(os.path.join(class_path, subdir))
                counter = 0
                for img in imgs:
                    i_path = os.path.join(class_path, subdir, img)
                    image_resized = self.trans_fname_img(i_path, self.data_size)

                    if subdir not in class2imgs_dict.keys():
                        class2imgs_dict[subdir] = []
                    class2imgs_dict[subdir].append(image_resized)
                    
                    counter += 1
                    if counter >= self.k1_clean:
                        break


        elif self.mode == 'addnoisy':
            class_i = 0
            for cid in part_cid:         # ./train/mode(train/test)/class_name(label)
                subdir = class_dirs[cid]
                imgs = os.listdir(os.path.join(class_path, subdir))
                counter = 0
                for img in imgs:
                    i_path = os.path.join(class_path, subdir, img)
                    image_resized = self.trans_fname_img(i_path, self.data_size)

                    if class_i not in class2imgs_dict.keys():
                        class2imgs_dict[class_i] = []
                    class2imgs_dict[class_i].append(image_resized)

                    counter += 1
                    if counter >= self.k1_clean:
                        break

                # t_path = os.path.join(self.root, 'cub-clean-noisy/noisy_500', str(cid + 1))
                t_path = os.path.join(self.root, 'cub-mix/noisy_500', str(cid + 1))

                imgs = os.listdir(t_path)
                #print(t_path)
                #imgs = sorted(imgs, key=lambda img : int(img.split('.')[0]))
                add_counter = 0
                for img in imgs:
                    i_path = os.path.join(t_path, img)
                    image_resized = self.trans_fname_img(i_path, self.data_size)

                    class2imgs_dict[class_i].append(image_resized)

                    add_counter += 1
                    if add_counter >= self.k2_add:
                        break

                #print(len(class2imgs_dict[class_i]))
                class_i += 1


        elif self.mode == 'subnoisy':

            print('------K1 clean: {}, k2_add: {}, threshold: {}-----'.format(self.k1_clean, self.k2_add, self.threshold))
            f = open(os.path.join(self.root, 'cleaned_noisy.json'), 'r')
            sub_noisy_imgfile = json.load(f)

            class_i = 0
            for cid in part_cid:         # ./train/mode(train/test)/class_name(label)
                subdir = class_dirs[cid]
                imgs = os.listdir(os.path.join(class_path, subdir))
                counter = 0
                for img in imgs:
                    i_path = os.path.join(class_path, subdir, img)
                    image_resized = self.trans_fname_img(i_path, self.data_size)

                    if class_i not in class2imgs_dict.keys():
                        class2imgs_dict[class_i] = []
                    class2imgs_dict[class_i].append(image_resized)

                    counter += 1
                    if counter >= self.k1_clean:
                        break

                # cleaned noisy images
                t_path = os.path.join(self.root, 'cub-mix/noisy_'+str(self.k2_add), subdir)
                
                # according to threshold
                imgs = os.listdir(t_path)
                for img in imgs:

                    cof = sub_noisy_imgfile[subdir][img] 
                    #print(self.threshold)
                    if cof < self.threshold:
                        continue

                    i_path = os.path.join(t_path, img)
                    image_resized = self.trans_fname_img(i_path, self.data_size)
                    class2imgs_dict[class_i].append(image_resized)
                
                '''
                # according to portion
                choosed = 0
                scored_imgs = sub_noisy_imgfile[subdir]
                sorted_imgs = sorted(scored_imgs, key=scored_imgs.__getitem__, reverse=True) 
                for img in sorted_imgs:

                    i_path = os.path.join(t_path, img)
                    image_resized = self.trans_fname_img(i_path, self.data_size)
                    class2imgs_dict[class_i].append(image_resized)

                    #print(img, scored_imgs[img])
                    choosed += 1
                    if choosed >= int(self.k2_add*0.6):
                        break
                '''
                #print(len(class2imgs_dict[class_i]))
                class_i += 1

            f.close()
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

        return [support_data, support_label, query_data, query_label, filenames]

