# encoding=utf-8
import argparse
import os
import random
import numpy as np
import torch
# ----------------------------------------------
from models.DD_GAN_V1 import DD_GAN_V1
# ----------------------------------------------

def parse_args():
    desc = "Pytorch DD_GAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    parser.add_argument('--batch_size', type=int, default=50, help='input batch size [default: 30]')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.0001]')
    parser.add_argument('--max_epoch', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
    parser.add_argument('--log_info', default='log_info.txt', help='log_info txt')
    parser.add_argument('--model_dir', help='model dir [default: None, must input]')
    parser.add_argument('--checkpoint_dir', default='checkpoint', help='Checkpoint dir [default: checkpoint]')
    parser.add_argument('--snapshot', type=int, default=20, help='how many epochs to save model')
    parser.add_argument('--network', default=None, help='which network model to be used')
    parser.add_argument('--sketch_lr', type=lambda x: utils.restricted_float(x, [1e-5, 0.5]), default=0.0001, metavar='LR', help='Initial learning rate [1e-5, 5e-4] (default: 1e-4)')
    parser.add_argument('--pretrain_model', default=None, help='use the pretrain model')
    return check_args(parser.parse_args())

def check_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def check_args(args):
    if args.model_dir is None:
        print('please create model dir')
        exit()
    if args.network is None:
        print('please select model!!!')
        exit()
    check_folder(args.checkpoint_dir)                                   # --checkpoint_dir
    check_folder(os.path.join(args.checkpoint_dir, args.model_dir))     # --chekcpoint_dir + model_dir 

    try: # --epoch
        assert args.max_epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')
    try: # --batch_size
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

def main():
    # args
    args = parse_args()
    if args is None: exit()
    args.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    # create model
    print('****************network: {}****************'.format(args.network))
    if args.network == 'DD_GAN_V1':
        gan = DD_GAN_V1(args)
 
    else:
        print('select model error!!!')
        exit()
    # exit()

    gan.build_model()
    # exit()

    if args.phase == 'train' :
        # cp mainly file to corresponding model dir

        os.system('cp main.py %s' % (os.path.join(args.checkpoint_dir, args.model_dir))) # bkp of main.py
        os.system('cp models/%s.py %s' % (args.network, os.path.join(args.checkpoint_dir, args.model_dir))) # bkp of model.py

        gan.train()
        print(" [*] Training finished!")
    # exit()
    if args.phase == 'test' :
        gan.test()
        print(" [*] Test finished!")

    if args.phase == 'extract' :
        gan.evalueonline()
        print(" [*] Test finished!")

    if args.phase == 'extract_cat':
        gan.extract_fea_concat()
        print(" [*] Extract feature finished")

if __name__ == '__main__':
    main()

