#!/usr/bin/env python
# pylint: disable=W0201
import sys
import os
import argparse
import numpy as np

from recogn_dataset import DataSet
from recogn_models import O3N
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from net.st_gcn import Model as GCN

class Processor():
    """
        Base Processor
    """

    def __init__(self, argv=None):

        self.load_args(argv)


    def load_args(self, argv=None):

        parser = self.get_parser()
        self.args = parser.parse_args(argv)


    def load_model(self):
        #self.model = O3N(self.args.model_type, self.args.num_label)
        self.model = GCN(3, self.args.num_label, {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}, True)
        #weights = torch.load(self.args.weights)
        #self.model.load_state_dict(weights)
        self.model = torch.nn.DataParallel(self.model).cuda()

    def load_weights(self):
        if self.args.weights:
            pretrained_dict = torch.load(self.args.weights)
            #model_dict = list(self.model.state_dict())[:-2]
            #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            self.model.load_state_dict(pretrained_dict)

    def init_weights(self, m):
        classname=m.__class__.__name__
        #print(classname)
        if classname.find('Conv2d') != -1 or classname.find('Conv1d') != -1:
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0) 
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


    def load_data(self):
        self.data_loader = dict()
        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=DataSet(self.args.train_list, self.args.train_label, self.args.num_select_frames),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_worker,
            drop_last=True)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=DataSet(self.args.test_list, self.args.test_label, self.args.num_select_frames),
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers=self.args.num_worker)
        

    def train(self):
        self.model.train()
        loader = self.data_loader['train']
        sum_acc = 0
        sum_loss = 0
        len = 0
        for data, label in loader:

            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()

            #print(data)

            #print(label)

            # forward
            output = self.model(data)
            #print(output)
            loss = self.loss(output, label)

            _, pre = torch.max(output, 1)
            #print(pre)
            #print(label)
            tmp_acc = pre.eq(label.view_as(pre)).sum().item() / pre.shape[0]
            sum_acc += tmp_acc
            sum_loss += loss.data
            len += 1


            # backward
            self.optimizer.zero_grad()

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_gradient)

            self.optimizer.step()

            # statistics
            #sys.stdout.write("acc: {} and loss:{}\n".format(acc, loss.data))
            #with open(self.args.save_output, "a") as f:
            #    f.write("acc: {} and loss:{}\n".format(tmp_acc, loss.data))
            #print(loss.data)
        with open(self.args.save_output, "a") as f:
            f.write("avg acc: {} and avg loss: {}\n".format(sum_acc/len, sum_loss/len))


    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        acc = 0
        len = 0
        for data, label in loader:
            
            # get data
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()

            # inference
            with torch.no_grad():
                output = self.model(data)

            _, pre = torch.max(output, 1)
            acc += pre.eq(label).sum().item() / pre.shape[0]
            len += 1
            #sys.stdout.write("test acc: {}\n".format(acc))
        with open(self.args.save_output, "a") as f:
            f.write("test acc: {}\n".format(acc/len))



            

            

    def start(self):

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in self.args.gpus)

        self.load_model()
        self.model.apply(self.init_weights)
        self.load_weights()
        self.load_data()

        #sys.stderr.write("train start:")
        # training phase
        if self.args.phase == 'train' or self.args.phase == "recogn":

            #self.optimizer = optim.SGD([{"params": self.model.module.gcn.st_gcn_networks.parameters()},
            #                           {"params": self.model.module.gcn.fcn.parameters(), "lr": 1e-1},
            #                          {"params": self.model.module.gcn.edge_importance.parameters(), "lr": 1e-1}], lr=1e-2, momentum=0.9, nesterov=True, weight_decay=0.0001)
            self.optimizer = optim.SGD(self.model.parameters(),lr=1e-2,momentum=0.9,nesterov=True,weight_decay=0.0001)
            self.loss = torch.nn.CrossEntropyLoss().cuda()
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)

            for epoch in range(self.args.start_epoch, self.args.num_epoch):

                # training
                self.train()

                self.scheduler.step()

                # save model
                if ((epoch + 1) % self.args.save_interval == 0) or (
                        epoch + 1 == self.args.num_epoch):
                    filename = self.args.save_weights.format(epoch + 1)
                    torch.save(self.model.state_dict(), filename)

                # evaluation
                if ((epoch + 1) % self.args.eval_interval == 0) or (
                        epoch + 1 == self.args.num_epoch):
                    self.test()
        # test phase
        elif self.args.phase == 'test':

            # the path of weights must be appointed
            if self.args.weights is None:
                raise ValueError('Please appoint --weights.')

            # evaluation
            self.test()

    @staticmethod
    def get_parser(add_help=False):

        #region argsuments yapf: disable
        # parameter priority: command line > config > default
        parser = argparse.ArgumentParser( add_help=add_help, description='Base Processor')

        parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')

        # processor
        parser.add_argument('--phase', default='recogn', help='must be train or test')
        parser.add_argument('--num_label', type=int, default=52, help='num_label')

        parser.add_argument('--save_interval', type=int, default=100, help='save_interval')
        parser.add_argument('--save_output', default='supervised/stdout_baseline', help='save_output')
        parser.add_argument('--save_weights', default='supervised/epoch{}_model_baseline.pt', help='save_weights')

        parser.add_argument('--eval_interval', type=int, default=5, help='eval_interval')
        parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
        parser.add_argument('--num_epoch', type=int, default=200, help='stop training in which epoch')

        parser.add_argument('--gpus', type=list, default=[0, 1], help='the indexes of GPUs for training or testing')
        parser.add_argument('--clip_gradient', type=int, default=100, help='clip_gradient')



        # dataset
        parser.add_argument('--root_file', default='', help='root file')
        parser.add_argument('--train_list', default='/mnt/Action2/linlilang/PKUMMD-Skeleton/PKUMMD_2/xview/L/train_data.npy', help='train list file')
        parser.add_argument('--test_list', default='/mnt/Action2/linlilang/PKUMMD-Skeleton/PKUMMD_2/xview/L/val_data.npy', help='test list file')
        parser.add_argument('--train_label', default='/mnt/Action2/linlilang/PKUMMD-Skeleton/PKUMMD_2/xview/L/train_label.pkl', help='train list file')
        parser.add_argument('--test_label', default='/mnt/Action2/linlilang/PKUMMD-Skeleton/PKUMMD_2/xview/L/val_label.pkl', help='test list file')



        parser.add_argument('--image_tmpl', default='{:05d}.jpg', help='image_tmpl')
        parser.add_argument('--num_select_frames', type=int, default=60, help='num_select_frames')

        parser.add_argument('--num_worker', type=int, default=4, help='the number of worker per gpu for data loader')
        parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
        parser.add_argument('--test_batch_size', type=int, default=4, help='test batch size')
        
        # models
        parser.add_argument('--weights', default=None, help='the weights for network initialization')
        parser.add_argument('--model_type', default='new', help='model_type')
        #endregion yapf: enable

        return parser

if __name__ == "__main__":
    processor = Processor()
    processor.start()
    
