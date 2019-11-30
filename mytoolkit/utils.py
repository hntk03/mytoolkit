# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd


class Standard():
    
    def __init__(self, train_dataloader, valid_dataloader, test_dataloader, model, optimizer, device, outfunc=False, num_classes=10, num_epochs=100):
        
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.optimizer = optimizer 
        self.device = device
        self.metrics_df = pd.DataFrame()
        self.outfunc = outfunc
        self.num_classes = num_classes
        self.status = None
        
    def loss(self):
        '''
        '''
        raise NotImplementedError()
        
    def _outfunc(self, x):
        return F.log_softmax(x, dim=1)
    
    def metrics(self, y, y_hat):
        '''
        '''
        pass
        
        
    def forward(self, batch, train):
        
        if train:
            self.model.train()
            optimizer.zero_grad()
            self.status = 'train_'
        else:
            self.model.eval()
            self.status = 'valid_'
            
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        # 出力層
        if self.outfunc:
            y_hat = self._outfunc(x)
        else:
            y_hat = self.model(x)
        
        loss = self.loss(y, y_hat)
        
        if train:
            loss.backward()
            optimizer.step()
            
        self.mesurement[self.status + 'loss'] += loss.item()
        
        self.metrics(y, y_hat, train)
        
    
    def _train(self):
        
        for batch in self.train_dataloader:
            self.forward(batch, train=True)
        self.mesurement[self.status + 'loss'] /= len(self.train_dataloader)
        
    def _valid(self):
        
        for batch in self.valid_dataloader:
            self.forward(batch, train=False)
        self.mesurement[self.status + 'loss'] /= len(self.valid_dataloader)
    
    def _test(self):
        net = self.net
        device = self.device
        net.eval()
        
        predicts = np.empty([0, self.num_classes])
        for x in self.test_loader:
            x = x.to(device)
            y_hat = net(inputs)
            y_hat = self._outfunc(y_hat).cpu().detach().numpy()
            
            predicts = np.vstack([predicts, outputs])

        return predicts
    
    def _print(self):
        dictionary = self.mesurement
        print_str = ''
        
        for d in dictionary:
            print_str = print_str + d[0] 
            print_str = ' ' + d[1] 
            
        print(print_str)
        
    
    def _epoch(self):
        
        
        for epochs_idx in range(self.num_epochs):
            
            self.mesurement = {'train_loss' : 0.0, 'valid_loss' : 0.0}
            
            self._train()
            self._valid() 
            
            self._print()
            
            self.metrics_df = self.metrics_df.append(self.mesurement, ignore_index=True)
        
    def run(self):
        self._epoch()


