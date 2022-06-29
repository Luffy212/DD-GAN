import torch
import torch.nn as nn
import torch.utils.data as data
import os
from torchvision import transforms,utils
from PIL import Image
import numpy as np
import cv2
import random

random.seed(0)
np.random.seed(0)

'''
In order to save training time, we first use ResNet with classification loss to extract features of the 3d shape projection and the sketch,
save them locally, and then the dataset directly reads the features and word embedding vectors for training.
We will post the preprocessed data later.

'''

def shape_load(x):
    with open(x, 'r') as f:
        lines = f.readlines()
    data = np.array([float(line.strip()) for line in lines], dtype=np.float32)
    return data
def sketch_load(path):
    img_pil = Image.open(path)
    img_pil = img_pil.resize((224,224))
    transform_pic1 = transforms.Compose([
        transforms.ToTensor(),
                                         ]
    )
    transform_pic2 = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                         ]
    )
    img_tensor = transform_pic1(img_pil)
    if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3,1,1)
    img_tensor = transform_pic2(img_tensor)
    return img_tensor

class Res_13fea_ZS_triplet_semgen(torch.utils.data.Dataset):
#---------------------------------------------
    def __init__(self,train = True):
        self.new_train_path = []
        self.label = {}
        self.train = train 
        if train == True:
            self.root = '/test/3D_retrieval/dataset/sketch_trainZS.txt'                      # sketch feature list path
            self.shape_root = '/test/3D_retrieval/dataset/shrec13_shapeMview_ZS_train.txt'   # shape feature list path
            self.class_root = '/test/data2/3d_shape/sketchShape/ZS_train_class.txt'          # label list path
        else:
            self.root = '/test/3D_retrieval/dataset/sketch_testZS.txt'
            self.shape_root = '/test/3D_retrieval/dataset/shrec13_shapeMview_ZS_test.txt'
            self.class_root = '/test/data2/3d_shape/sketchShape/ZS_test_class.txt'
            
            
        with open(self.root) as f:
            self.new_train_path = f.read().splitlines()
        with open(self.shape_root,'r') as fp:
            self.image_path = fp.read().splitlines()
        with open(self.class_root,'r') as fp:
            self.class_list = fp.read().splitlines()

        for i in range(len(self.class_list)):
            self.label[self.class_list[i]] = i
        
    def __getitem__(self,index):
        word_vec_path = '/test/data2/3d_shape/sketchShape/dataset/word_embedding_vector/'  # word embedding root
        fn = self.new_train_path[index].split(' ')
        sketch_path = fn[0]
        label = int(fn[1])
        cls = sketch_path.split('/')[-2]
        img_oral1 = shape_load(sketch_path)
        if self.train == True:
            #positive sample
            positive_classes = [v for v in self.image_path if '/' + cls + '/' in v]
            pos =np.random.choice(positive_classes,1)[0]
            shape_pos = shape_load(pos)
            #negative sample
            possible_classes = [x for x in self.class_list if x != cls]
            k =np.random.choice(possible_classes,1)[0]
            tmp = [v for v in self.image_path if '/' + k + '/' in v]
            neg =np.random.choice(tmp,1)[0]
            shape_neg = shape_load(neg)
            neg_label = self.label[k]

            return img_oral1,shape_pos,shape_neg,label,neg_label,np.load(word_vec_path + cls + '.npy'),np.load(word_vec_path + k + '.npy')
        else:
            return img_oral1,label
        
    def __len__(self):
        
        return len(self.new_train_path)

class Res_13Shapefea_ZS(data.Dataset):
    def __init__(self,mode='res'):
        
        self.new_train_path = []
        self.label = {}
        self.path_list = []
        self.root = '/test/3D_retrieval/dataset/shrec13_shapeMview_ZS_test.txt'
        self.class_root = '/test/data2/3d_shape/sketchShape/ZS_test_class.txt'
        with open(self.root) as f:
            self.new_train_path = f.read().splitlines()
        with open(self.class_root) as f:
            self.cls_ = f.read().splitlines()
        for i in range(len(self.cls_)):
            self.label[self.cls_[i]] = 79 + i
        

    def __getitem__(self,index):
        word_vec_path = '/test/data2/3d_shape/sketchShape/dataset/word_embedding_vector/'
        fn = self.new_train_path[index]
        img_oral = shape_load(fn)
        cls = fn.split('/')[-2]
        label_cls = self.label[cls]
        return img_oral,label_cls
        
    def __len__(self):
        return len(self.new_train_path)
