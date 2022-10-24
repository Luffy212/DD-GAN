import torch
import torch.nn as nn
import torch.utils.data as data
import os
from torchvision import transforms,utils
from PIL import Image
import numpy as np
import random

random.seed(0)
np.random.seed(0)


def txt_load(x):
    with open(x, 'r') as f:
        lines = f.readlines()
    data = np.array([float(line.strip()) for line in lines], dtype=np.float32)
    return data


class Res_13fea_ZS_triplet_semgen(torch.utils.data.Dataset):
    def __init__(self,train = True):
        self.new_train_path = []
        self.label = {}
        self.train = train 
        if train == True:
            self.root = './datasets/sketch_trainZS.txt'
            self.shape_root = './datasets/shrec13_shapeMview_ZS_train.txt'
            self.class_root = './datasets/ZS_train_class.txt'
        else:
            self.root = './datasets/sketch_testZS.txt'
            self.root2 = './datasets/sketch_trainZS.txt'
            self.shape_root = './datasets/shrec13_shapeMview_ZS_test.txt'
            self.class_root = './datasets/ZS_test_class.txt'
            with open(self.root2) as f:
                self.new_train_path2 = f.read().splitlines()

        with open(self.root) as f:
            self.new_train_path = f.read().splitlines()
        with open(self.shape_root,'r') as fp:
            self.image_path = fp.read().splitlines()
        with open(self.class_root,'r') as fp:
            self.class_list = fp.read().splitlines()

        for i in range(len(self.class_list)):
            self.label[self.class_list[i]] = i

    
    def __getitem__(self,index):
        word_vec_path = './datasets/word_embedding_vector/'
        fn = self.new_train_path[index].split(' ')
        sketch_path = fn[0]
        label = int(fn[1])
        cls = sketch_path.split('/')[-2]
        img_oral1 = txt_load(sketch_path)
        if self.train == True:
            #positive sample
            positive_classes = [v for v in self.image_path if '/' + cls + '/' in v]
            pos =np.random.choice(positive_classes,1)[0]
            shape_pos = txt_load(pos)
            #negative sample
            possible_classes = [x for x in self.class_list if x != cls]
            k =np.random.choice(possible_classes,1)[0]
            tmp = [v for v in self.image_path if '/' + k + '/' in v]
            neg =np.random.choice(tmp,1)[0]
            shape_neg = txt_load(neg)
            neg_label = self.label[k]

            return img_oral1,shape_pos,shape_neg,label,np.load(word_vec_path + cls + '.npy'),np.load(word_vec_path + k + '.npy')
        else: 
            sketch_path2 =np.random.choice(self.new_train_path2,1)[0]
            img_oral = txt_load(sketch_path2.split(' ')[0])
            return img_oral1,img_oral,label,np.load(word_vec_path + cls + '.npy')
        
    def __len__(self):
        
        return len(self.new_train_path)


class Res_13Shapefea_ZS(data.Dataset):
    def __init__(self):
        
        self.new_train_path = []
        self.label = {}
        self.path_list = []
        self.root = './datasets/shrec13_shapeMview_ZS_test.txt'
        self.root2 = './datasets/shrec13_shapeMview_ZS_train.txt'
        self.class_root = './datasets/ZS_test_class.txt'
        with open(self.root) as f:
            self.new_train_path = f.read().splitlines()
        with open(self.root2) as f:
            self.new_train_path2 = f.read().splitlines()
        with open(self.class_root) as f:
            self.cls_ = f.read().splitlines()
        for i in range(len(self.cls_)):
            self.label[self.cls_[i]] = 79 + i

    def __getitem__(self,index):
        word_vec_path = './datasets/dataset/word_embedding_vector/'
        fn = self.new_train_path[index]
        fn2 = self.new_train_path2[index]
        img_oral = txt_load(fn)
        img_oral2 = txt_load(fn2)
        cls = fn.split('/')[-2]
        label_cls = self.label[cls]
        return img_oral,img_oral2,label_cls,np.load(word_vec_path + cls + '.npy')
        
    def __len__(self):
        return len(self.new_train_path)
