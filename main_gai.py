# coding: utf-8
import os
import time
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from torch.autograd import Variable
from lr_scheduler import *
from model import *
from dataset import *

import tensorboardX
import tensorflow as tf
from tensorboardX import SummaryWriter

if (__name__ == '__main__'):    
    SEED = 1
    torch.manual_seed(SEED) #保证随机初始化的状态相同
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    opt = __import__('option')


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def data_loader(data_path, batch_size, workers):
    dsets = {x: MyDataset(x, data_path) for x in ['train', 'val', 'test']}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size, shuffle=True, num_workers=workers) for x in ['train', 'val', 'test']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
    print('\nStatistics: train: {}, val: {}, test: {}'.format(dset_sizes['train'], dset_sizes['val'], dset_sizes['test']))
    return dset_loaders, dset_sizes

dset_loaders, datasets = data_loader(opt.dataset, opt.batch_size, opt.workers)

if (__name__ == '__main__'):
    model = lipreading(mode=opt.mode, inputDim=512, hiddenDim=512, nClasses=opt.nClasses, frameLen=29, every_frame=opt.every_frame)

    writer = SummaryWriter()

    if (hasattr(opt, 'weights')):
        pretrained_dict = torch.load(opt.weights)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    train_loader = dset_loaders['train']
    val_loader = dset_loaders['val']
    tst_loader = dset_loaders['test']
    
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-5)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

    iteration = 0
    n = 0

    for epoch in range(opt.epochs):
        start_time = time.time()
        exp_lr_scheduler.step()

        for i, (inputs, targets) in enumerate(train_loader):
            #print(i)
            inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
            outputs = model(inputs)
            #print(outputs.size())
            
            #if opt.every_frame:
            #    outputs = torch.mean(outputs, 1)

            #print(outputs.size(), targets.size())
            loss = criterion(outputs, targets)
            optimizer.zero_grad()

            writer.add_scalar('data/scalar_loss_ce', loss.detach().cpu().numpy(), iteration)

            iteration += 1

            loss.backward()
            optimizer.step()

            train_loss = loss.item()

            print('iteration:%d, epoch:%d, train_loss:%.6f'%(iteration, epoch, train_loss))

            if (iteration % 2000) == 0:
                corrects = 0
                all_data = 0
                with torch.no_grad():
                    for idx, (inputs, targets) in enumerate(val_loader):
                        inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
                        outputs = model(inputs)
                        #if opt.every_frame:
                        #    outputs = torch.mean(outputs, 1)
                        _, preds = torch.max(F.softmax(outputs, dim=1).data, 1)
                        corrects += torch.sum(preds == targets.data)
                        #print(corrects)
                        all_data += len(inputs)
                        #print(all_data)
                        #print('acc=',corrects.cpu().numpy() / all_data)
                acc = corrects.cpu().numpy() / all_data
            
                writer.add_scalar('data/scalar_acc', acc, n)
                print('iteration:%d, epoch:%d, acc:%d' % (iteration, epoch, acc))

                savename = os.path.join(opt.savedir, opt.mode+'_iteration_{}_epoch_{}_acc_{}.pt'.format(iteration, epoch, acc))
                savepath = os.path.split(savename)[0]
                if (not os.path.exists(savepath)):os.makedirs(savepath)
        
                if acc >= 0.80: torch.save(model.state_dict(), savename)
                n += 1
    
    writer.close()

        
    