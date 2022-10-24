# encoding=utf-8

import numpy as np
import sys
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from dataset import Res_13Shapefea_ZS,Res_13fea_ZS_triplet_semgen
from utils.load_label import *
from utils.RetrievalEvaluation import RetrievalEvaluation
import torch.nn.functional as F
import sklearn.metrics as metrics

class DD_GAN(object):
    def __init__(self,args):
        self.model_name = args.network
        self.workers = args.workers
        self.checkpoint_dir = args.checkpoint_dir
        self.model_dir = args.model_dir
        self.pretrain_model_Shape = args.pretrain_model
        self.epoch = args.max_epoch
        self.batch_size = args.batch_size
        self.snapshot = args.snapshot  
        self.lr = args.lr

        self.sketch_dataset = Res_13fea_ZS_triplet_semgen()
        self.sketch_dataloader = torch.utils.data.DataLoader(self.sketch_dataset,batch_size = self.batch_size,shuffle=True, num_workers= 4)
        self.num_batches = len(self.sketch_dataset) // self.batch_size
        if args.phase == 'train':
            print('training...')
            self.log_info = args.log_info
            self.LOG_FOUT = open(os.path.join(self.checkpoint_dir, self.model_dir, self.log_info), 'w')
            self.LOG_FOUT.write(str(args)+'\n')
        elif args.phase == 'test':
            print('testing...')

        cudnn.benchmark = True  # cudnn

    def build_model(self):
        self.device = self._get_device()
        self.gen_skecon = Generator_content(
            in_dim=2048, out_dim=300, noise=False, use_dropout=True).cuda()
        # Sketch style generator
        self.gen_skesty = Generator_style(
            in_dim=2048, out_dim=300, noise=False, use_dropout=True).cuda()
        # Image context generator
        self.gen_imgcon = Generator_content(
            in_dim=2048, out_dim=300, noise=False, use_dropout=True).cuda()
        # Image style generator
        self.gen_imgsty = Generator_style(
            in_dim=2048, out_dim=300, noise=False, use_dropout=True).cuda()

        # Sketch decoder
        self.skedec = Generator(in_dim=600, out_dim=2048,
                                noise=False, use_dropout=False).cuda()
        # Image decoder
        self.imgdec = Generator(in_dim=600, out_dim=2048,
                                noise=False, use_dropout=False).cuda()
        # Sketch discriminator
        self.disc_ske = Discriminator(in_dim=2048, noise=True, use_batchnorm=True).cuda()
        # Image discriminator
        self.disc_img = Discriminator(in_dim=2048, noise=True, use_batchnorm=True).cuda()

        self.semantic_sk = Semantic_fc(in_dim = 300 , out_dim = 300).cuda()
        self.semantic_im = Semantic_fc(in_dim = 300 , out_dim = 300).cuda()

        self.optimizer_gen = optim.Adam(list(self.gen_skecon.parameters()) + list(self.gen_skesty.parameters()) +
                                        list(self.gen_imgcon.parameters()) + list(self.gen_imgsty.parameters()) +
                                        list(self.skedec.parameters()) + list(self.semantic_im.parameters()) + 
                                        list(self.semantic_sk.parameters()) + list(self.imgdec.parameters()),
                                        lr=self.lr, weight_decay=(10e-5))

        self.optimizer_disc = optim.SGD(list(self.disc_ske.parameters()) + list(self.disc_img.parameters()),
                                        lr=self.lr, momentum= 0.9)

        self.loss_fn = nn.CrossEntropyLoss()
        self.criterion_reg = nn.MSELoss()
        self.criterion_cyc = nn.MSELoss()
        self.criterion_gan = GANLoss(use_lsgan=True)

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    
    def contrastiveLoss(self,input_fea_1, input_fea_2, input_fea_3, margin):
        distance_positive = F.pairwise_distance(input_fea_1,input_fea_2)
        distance_positive2 = F.pairwise_distance(input_fea_1,input_fea_3)
        distance_negative = torch.max(torch.zeros_like(distance_positive2), margin - distance_positive2)

        distance_contrastive = distance_positive +  distance_negative
        distance_loss = torch.mean(distance_contrastive)

        return distance_loss

    def cosine_similarity(self, input, target):
        cosine_sim = input.unsqueeze(1).bmm(target.unsqueeze(2)).squeeze()
        norm_i = input.norm(p=2, dim=1)
        norm_t = target.norm(p=2, dim=1)
        return cosine_sim / (norm_i * norm_t)

    def cosine_loss(self,input, target):
        cosine_sim = self.cosine_similarity(input, target)
        cosine_dist = (1 - cosine_sim) / 2
        return cosine_dist

    def train(self):

        could_load, save_epoch = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = save_epoch
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 1
            print(" [!] start epoch: {}".format(start_epoch))

        # loop for epoch
        start_time = time.time()
        best_mAP = 0.
        for epoch in range(start_epoch, self.epoch+1):
            for idx, datas in enumerate(self.sketch_dataloader, 0):
                if idx+1 > self.num_batches: continue
                sketch_oral,shape_oral,shape_neg_oral,sketch_label,sketch_w2v,shape_neg_w2v = datas
                
                # print("shape size:",shape_oral.shape)

                sketch_oral_1 = sketch_oral.float().cuda()
                shape_oral_1 = shape_oral.float().cuda()
                shape_neg_oral_1 = shape_neg_oral.float().cuda()
                sketch_label_tensor_1 = torch.from_numpy(np.array(list(sketch_label)))

                # extract sketch content and style feature
                sketch_fea_oral_1,sketch_con_cls = self.gen_skecon(sketch_oral_1)
                sketch_sty, _ = self.gen_skesty(sketch_oral_1)

                # extract sketch content and style feature
                shape_fea_oral_1,shape_con_cls = self.gen_imgcon(shape_oral_1)
                image_sty, _ = self.gen_imgsty(shape_oral_1)
                shape_fea_oral_neg_1,shape_con_neg_cls = self.gen_imgcon(shape_neg_oral_1)
                image_neg_sty, _ = self.gen_imgsty(shape_neg_oral_1)


                # map the visual feature to semantic space
                sketch_sem = self.semantic_sk(sketch_fea_oral_1)
                shape_sem = self.semantic_im(shape_fea_oral_1)
                shape_neg_sem = self.semantic_im(shape_fea_oral_neg_1)

                # cross domain generation
                ske2img = self.imgdec(
                    torch.cat((sketch_sem, image_sty), 1))
                ske2img2 = self.imgdec(
                    torch.cat((sketch_sem, image_neg_sty), 1))
                img2ske = self.skedec(
                    torch.cat((shape_fea_oral_1, sketch_sty), 1))

                # self domain reconstruction
                sketch_recon = self.imgdec(
                    torch.cat((sketch_sem, sketch_sty), 1))
                image_recon = self.imgdec(
                    torch.cat((shape_sem, image_sty), 1))
                image_neg_recon = self.skedec(
                    torch.cat((shape_neg_sem, image_neg_sty), 1))

                shape_fea_fake,shape_fake_cls = self.gen_imgcon(ske2img)
                shape_fea_fake_neg_1,shape_fake_neg_cls = self.gen_imgcon(ske2img2)

                shape_fake_sem = self.semantic_im(shape_fea_fake)
                shape_neg_fake_sem = self.semantic_im(shape_fea_fake_neg_1)
                # #========================== Train D ==========================================
                loss_disc_adv = self.criterion_gan(self.disc_img(ske2img), False) + self.criterion_gan(self.disc_img(ske2img2), False) + \
                                self.criterion_gan(self.disc_ske(img2ske), False) + \
                                self.criterion_gan(self.disc_img(shape_oral_1), True) + self.criterion_gan(self.disc_ske(sketch_oral_1), True)
   

                loss_D = loss_disc_adv
                self.optimizer_disc.zero_grad()
                loss_D.backward(retain_graph= True)
                self.optimizer_disc.step()
                #============================= Train G ===========================================
                _,pred_true_sketch1 = sketch_con_cls.topk(1)
                train_pred_sketch1 = (pred_true_sketch1.detach().cpu().numpy())
                train_true_sketch1 = (sketch_label_tensor_1.cpu().numpy())
                acc_sketch1 = metrics.accuracy_score(train_true_sketch1, train_pred_sketch1)
                _,pred_true_shape1 = shape_con_cls.topk(1)
                train_pred_shape1 = (pred_true_shape1.detach().cpu().numpy())
                train_true_shape1 = (sketch_label_tensor_1.cpu().numpy())
                acc_shape1 = metrics.accuracy_score(train_true_shape1, train_pred_shape1)
                
                loss_gen_adv = self.criterion_gan(self.disc_img(ske2img), True) + self.criterion_gan(self.disc_img(ske2img2), True) + \
                                self.criterion_gan(self.disc_ske(img2ske), True)
                cls_loss = self.loss_fn(sketch_con_cls,sketch_label_tensor_1.cuda()) + self.loss_fn(shape_con_cls,sketch_label_tensor_1.cuda()) + \
                self.loss_fn(shape_fake_cls,sketch_label_tensor_1.cuda()) + self.loss_fn(shape_fea_fake_neg_1,sketch_label_tensor_1.cuda())
                cross_loss_1 = self.contrastiveLoss(sketch_fea_oral_1,shape_fea_oral_1,shape_fea_oral_neg_1, 20)
                sem_loss = self.cosine_loss(sketch_sem,sketch_w2v.cuda()) + self.cosine_loss(shape_sem,sketch_w2v.cuda()) \
                 + self.cosine_loss(shape_neg_sem,shape_neg_w2v.cuda()) + self.cosine_loss(shape_fake_sem,sketch_w2v.cuda()) \
                 + self.cosine_loss(shape_neg_fake_sem,sketch_w2v.cuda())
                loss_sem = sem_loss.mean()
                loss_gen_cyc = self.criterion_cyc(image_recon, shape_oral_1) + self.criterion_cyc(sketch_recon, sketch_oral_1) \
                    + self.criterion_cyc(image_neg_recon, shape_neg_oral_1)

                self.loss = 2 * cross_loss_1 + cls_loss + loss_sem + 0.1 * loss_gen_adv + 2 * loss_gen_cyc
                
                self.optimizer_gen.zero_grad()
                self.loss.backward()
                self.optimizer_gen.step()
                if idx % 50 ==0:
                    print("Epoch: [%2d] [%4d/%4d] time: %2dm %2ds loss: %.4f sim_loss: %.4f acc1: %.4f acc2: %.4f"  \
                              % (epoch, idx+1, self.num_batches, (time.time()-start_time)/60,(time.time()-start_time)%60,self.loss.item(),loss_D.item(),acc_sketch1,acc_shape1))
                    self.log_string("Epoch: [%2d] [%4d/%4d] time: %2dm %2ds loss: %.4f sim_loss: %.4f acc1: %.4f acc2: %.4f" \
                              % (epoch, idx+1, self.num_batches, (time.time()-start_time)/60,(time.time()-start_time)%60,self.loss.item(),loss_D.item(),acc_sketch1,acc_shape1))
                    
            if epoch % 1 == 0:
                with torch.no_grad():
                    mAP = self.evalueonline()
                    is_best = mAP > best_mAP
                    if is_best:
                        best_mAP = mAP
                        self.save(self.checkpoint_dir, 'best',self.epoch)
                    self.log_string("MAP:%s"%(mAP))
            self.save(self.checkpoint_dir, 'last', self.epoch)
        self.LOG_FOUT.close()

    def test(self):
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")
        self.gen_skecon.eval()
        self.gen_imgcon.eval()
        sketchLabMatrix = []
        shapeLabMatrix = []
        sketchMatrix = []
        shapeMatrix = []
        start_time = time.time()

        sketch_dataset = Res_13fea_ZS_triplet_semgen(train = False)
        sketch_dataloader = torch.utils.data.DataLoader(sketch_dataset,batch_size = 4,shuffle=True, num_workers= 4)
        shape_proj_dataset = Res_13Shapefea_ZS()
        shape_dataloader = torch.utils.data.DataLoader(shape_proj_dataset,batch_size = 4,shuffle = True,num_workers = 4)
        for idx,datas in enumerate(sketch_dataloader,0):
            sketch_oral,_,sketch_label,_ = datas
            sketchs = Variable(sketch_oral).float().cuda()
            sketch_fea_tmp,_ = self.gen_skecon(sketchs.cuda())
            sketchMatrix.append(sketch_fea_tmp)
            sketchLabMatrix.append(sketch_label)

        for idx, datas2 in enumerate(shape_dataloader, 0):
            shape_oral,_,shape_label,_ = datas2
            shape_oral_tensor = Variable(shape_oral).float().cuda()  
            shape_fea_tmp,_ = self.gen_imgcon(shape_oral_tensor)

            shapeLabMatrix.append(shape_label)
            shapeMatrix.append(shape_fea_tmp)

        sketchMatrix = torch.cat(sketchMatrix)
        sketchLabMatrix = torch.cat(sketchLabMatrix).numpy()
        shapeMatrix = torch.cat(shapeMatrix)
        shapeLabMatrix = torch.cat(shapeLabMatrix).numpy()

        print('shapeMatrix:',shapeMatrix.shape)
        print('sketchMatrix:',sketchMatrix.shape)
        distM=1 - torch.mm(F.normalize(sketchMatrix, dim=-1),
            F.normalize(shapeMatrix, dim=-1).t())
        distM = distM.detach().cpu().numpy()
        sketch_test_label = np.array(sketchLabMatrix).flatten()
        shape_test_label = np.array(shapeLabMatrix).flatten()
        C_depths = retrievalParamSP_v2(shape_test_label,sketch_test_label)
        model_label = np.array(shape_test_label).astype(int)
        test_label = np.array(sketch_test_label).astype(int)
        C_depths = C_depths.astype(int)
        # print(C_depths.shape,test_label.shape,model_label.shape,distM.shape)
        nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec, rankArray = RetrievalEvaluation(C_depths, distM,
                                                                                                     model_label,
                                                                                                     test_label,
                                                                                                     testMode=1)
        print('The NN is %5f' % (nn_av))
        print('The FT is %5f' % (ft_av))
        print('The ST is %5f' % (st_av))
        print('The DCG is %5f' % (dcg_av))
        print('The E is %5f' % (e_av))
        print('The MAP is %5f' % (map_))

    def evalueonline(self):
        self.gen_skecon.eval()
        self.gen_imgcon.eval()
        sketchLabMatrix = []
        shapeLabMatrix = []
        sketchMatrix = []
        shapeMatrix = []
        start_time = time.time()

        sketch_dataset = Res_13fea_ZS_triplet_semgen(train = False)
        sketch_dataloader = torch.utils.data.DataLoader(sketch_dataset,batch_size = 4,shuffle=True, num_workers= 4)
        shape_proj_dataset = Res_13Shapefea_ZS()
        shape_dataloader = torch.utils.data.DataLoader(shape_proj_dataset,batch_size = 4,shuffle = True,num_workers = 4)
        for idx,datas in enumerate(sketch_dataloader,0):
            sketch_oral,_,sketch_label,_ = datas
            sketchs = Variable(sketch_oral).float().cuda()
            sketch_fea_tmp,_ = self.gen_skecon(sketchs.cuda())
            sketchMatrix.append(sketch_fea_tmp)
            sketchLabMatrix.append(sketch_label)

        for idx, datas2 in enumerate(shape_dataloader, 0):
            shape_oral,_,shape_label,_ = datas2
            shape_oral_tensor = Variable(shape_oral).float().cuda()  
            shape_fea_tmp,_ = self.gen_imgcon(shape_oral_tensor)
            shapeLabMatrix.append(shape_label)
            shapeMatrix.append(shape_fea_tmp)

        sketchMatrix = torch.cat(sketchMatrix)
        sketchLabMatrix = torch.cat(sketchLabMatrix).numpy()
        shapeMatrix = torch.cat(shapeMatrix)
        shapeLabMatrix = torch.cat(shapeLabMatrix).numpy()

        print('shapeMatrix:',shapeMatrix.shape)
        print('sketchMatrix:',sketchMatrix.shape)
        distM=1 - torch.mm(F.normalize(sketchMatrix, dim=-1),
            F.normalize(shapeMatrix, dim=-1).t())
        distM = distM.detach().cpu().numpy()
        sketch_test_label = np.array(sketchLabMatrix).flatten()
        shape_test_label = np.array(shapeLabMatrix).flatten()
        C_depths = retrievalParamSP_v2(shape_test_label,sketch_test_label)
        model_label = np.array(shape_test_label).astype(int)
        test_label = np.array(sketch_test_label).astype(int)
        C_depths = C_depths.astype(int)
        print(C_depths.shape,test_label.shape,model_label.shape,distM.shape)
        nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec, rankArray = RetrievalEvaluation(C_depths, distM,
                                                                                                     model_label,
                                                                                                     test_label,
                                                                                                     testMode=1)
        print('The NN is %5f' % (nn_av))
        print('The FT is %5f' % (ft_av))
        print('The ST is %5f' % (st_av))
        print('The DCG is %5f' % (dcg_av))
        print('The E is %5f' % (e_av))
        print('The MAP is %5f' % (map_))
        self.gen_skecon.train()
        self.gen_imgcon.train()

        return map_

    def log_string(self, out_str):
        self.LOG_FOUT.write(out_str+'\n')
        self.LOG_FOUT.flush()


    def load(self, checkpoint_dir):
        if self.pretrain_model_Shape is None:
            print('################ new training ################')
            return False, 1

        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(
            checkpoint_dir, self.model_dir, self.model_name)

        # ----------------- load D -------------------
        if not self.pretrain_model_Shape is None:
            resume_file_D = os.path.join(
                checkpoint_dir, self.pretrain_model_Shape)
            flag_D = os.path.isfile(resume_file_D)
            if flag_D == False:
                print('ShapeNet--> Error: no checkpoint directory found!')
                exit()
            else:
                print('resume_file_ShapeNet------>: {}'.format(resume_file_D))
                checkpoint = torch.load(resume_file_D)

                self.optimizer_gen.load_state_dict(checkpoint['optimizer_gen'])
                self.optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
                self.gen_skecon.load_state_dict(checkpoint['gen_skecon'])
                self.gen_skesty.load_state_dict(checkpoint['gen_skesty'])
                self.gen_imgcon.load_state_dict(checkpoint['gen_imgcon'])
                self.gen_imgsty.load_state_dict(checkpoint['gen_imgsty'])
                self.skedec.load_state_dict(checkpoint['skedec'])
                self.imgdec.load_state_dict(checkpoint['imgdec'])
                D_epoch = checkpoint['gen_epoch']
        else:
            print(" [*] Failed to find the pretrain_model_Shape")
            exit()

        print(" [*] Success to load model --> {}".format(self.pretrain_model_Shape))
        return True, D_epoch

    def save(self, checkpoint_dir, sname, index_epoch):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        save_name = str(sname)+'_'+"full"
        path_save_D = os.path.join(checkpoint_dir, save_name+'_P.pth')
        # print('Save Path for G: {}'.format(path_save_G))
        print('Save Path for D: {}'.format(path_save_D))
        torch.save({
            'optimizer_gen': self.optimizer_gen.state_dict(),
            'optimizer_disc': self.optimizer_disc.state_dict(),
            'gen_epoch': index_epoch,
            'gen_skecon': self.gen_skecon.state_dict(),
            'gen_skesty': self.gen_skesty.state_dict(),
            'gen_imgcon': self.gen_imgcon.state_dict(),
            'gen_imgsty': self.gen_imgsty.state_dict(),
            'skedec': self.skedec.state_dict(),
            'imgdec': self.imgdec.state_dict(),
            'disc_ske': self.disc_ske.state_dict(),
            'disc_img': self.disc_img.state_dict()
        }, path_save_D)

    def MSE_LOSS(self, label, pred):
        return tf.losses.mean_squared_error(label, pred)




################################################################################################
# -------------------------------- class of nework structure -----------------------------------
################################################################################################

class Generator_content(nn.Module):
    def __init__(self, in_dim=4096, out_dim=300, noise=True, use_batchnorm=True, use_dropout=False):
        super(Generator_content, self).__init__()
        hid_dim = int((in_dim + out_dim) / 2)
        modules = list()
        modules.append(nn.Linear(in_dim, hid_dim))
        if use_batchnorm:
            modules.append(nn.BatchNorm1d(hid_dim))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        if noise:
            modules.append(GaussianNoiseLayer(mean=0.0, std=0.2))
        if use_dropout:
            modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(hid_dim, hid_dim))
        if use_batchnorm:
            modules.append(nn.BatchNorm1d(hid_dim))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        if noise:
            modules.append(GaussianNoiseLayer(mean=0.0, std=0.2))
        if use_dropout:
            modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(hid_dim, out_dim))

        self.gen = nn.Sequential(*modules)
        self.cls_fc = nn.Linear(300,90)

    def forward(self, x):
        x_fea = self.gen(x)
        cls_fea = self.cls_fc(x_fea)
        return x_fea,cls_fea

class GaussianNoiseLayer(nn.Module):    
    def __init__(self, mean=0.0, std=0.2):
        super(GaussianNoiseLayer, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.training:
            noise = x.data.new(x.size()).normal_(self.mean, self.std)
            if x.is_cuda:
                noise = noise.cuda()
            x = x + noise
        return x

class Generator(nn.Module):
    def __init__(self, in_dim=2048, out_dim=300, noise=True, use_batchnorm=True, use_dropout=False):
        super(Generator, self).__init__()
        hid_dim = int((in_dim + out_dim) / 2)
        modules = list()
        modules.append(nn.Linear(in_dim, hid_dim))
        if use_batchnorm:
            modules.append(nn.BatchNorm1d(hid_dim))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        if noise:
            modules.append(GaussianNoiseLayer(mean=0.0, std=0.2))
        if use_dropout:
            modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(hid_dim, hid_dim))
        if use_batchnorm:
            modules.append(nn.BatchNorm1d(hid_dim))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        if noise:
            modules.append(GaussianNoiseLayer(mean=0.0, std=0.2))
        if use_dropout:
            modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(hid_dim, out_dim))

        self.gen = nn.Sequential(*modules)

    def forward(self, x):
        return self.gen(x)

class Generator_style(nn.Module):
    def __init__(self, in_dim=4096, out_dim=300, noise=True, use_batchnorm=True, use_dropout=False):
        super(Generator_style, self).__init__()
        hid_dim = int((in_dim + out_dim) / 2)
        modules = list()
        modules.append(nn.Linear(in_dim, hid_dim))
        if use_batchnorm:
            modules.append(nn.BatchNorm1d(hid_dim))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        if noise:
            modules.append(GaussianNoiseLayer(mean=0.0, std=0.2))
        if use_dropout:
            modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(hid_dim, hid_dim))
        if use_batchnorm:
            modules.append(nn.BatchNorm1d(hid_dim))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        if noise:
            modules.append(GaussianNoiseLayer(mean=0.0, std=0.2))
        if use_dropout:
            modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(hid_dim, out_dim))

        self.gen = nn.Sequential(*modules)
        self.cls_fc = nn.Linear(300, 2)

    def forward(self, x):
        x_fea = self.gen(x)
        cls_fea = self.cls_fc(x_fea)
        return x_fea, cls_fea


class Semantic_fc(nn.Module):
    def __init__(self, in_dim = 300, out_dim = 300):
        super(Semantic_fc,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim,300),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(300, out_dim),
            )
    def forward(self,x):
        y = self.fc(x)
        return y

class Discriminator(nn.Module):
    def __init__(self, in_dim=300, out_dim=1, noise=True, use_batchnorm=True, use_dropout=False,
                 use_sigmoid=False):
        super(Discriminator, self).__init__()
        hid_dim = int(in_dim / 2)
        modules = list()
        if noise:
            modules.append(GaussianNoiseLayer(mean=0.0, std=0.3))
        modules.append(nn.Linear(in_dim, hid_dim))
        if use_batchnorm:
            modules.append(nn.BatchNorm1d(hid_dim))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        if use_dropout:
            modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(hid_dim, hid_dim))
        if use_batchnorm:
            modules.append(nn.BatchNorm1d(hid_dim))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        if use_dropout:
            modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(hid_dim, out_dim))
        if use_sigmoid:
            modules.append(nn.Sigmoid())

        self.disc = nn.Sequential(*modules)

    def forward(self, x):
        return self.disc(x)

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True):
        super(GANLoss, self).__init__()
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        # Get soft and noisy labels
        if target_is_real:
            target_tensor = 0.7 + 0.3 * torch.rand(input.size(0))
        else:
            target_tensor = 0.3 * torch.rand(input.size(0))
        if input.is_cuda:
            target_tensor = target_tensor.cuda()
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input.squeeze(), target_tensor)

