import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class FreqMask(object):
    
    ''' 
    https://arxiv.org/abs/1904.08779
    input shape : H, W, C
    mask : mask value --> mean  or zero
    ''' 

    def __init__(self, F=30, mask='mean'):
        self.F = F
        assert (mask == 'mean' or mask == 'zero'), 'can not replace value'
        self.mask = mask

    def __call__(self, data):
        
        data = data.copy()
        upsilon = data.shape[1]
        f = np.random.randint(0, self.F)
        f0 = np.random.randint(0, upsilon-f)
        
        if (f0 == f0 + f): return data

        fmax = np.random.randint(f0, f0+f)
        
        if self.mask == 'mean':
            mean = data.mean()
            data[f0:fmax, :, :] = mean
        elif self.mask == 'zero':
            data[f0:fmax, :, :] = 0
            
        return data



class TimeMask(object):
    
    ''' 
    https://arxiv.org/abs/1904.08779
    input shape : H, W, C
    mask : mask value --> mean  or zero
    ''' 

    def __init__(self, T=30, mask='mean'):
        self.T = T 
        assert (mask == 'mean' or mask == 'zero'), 'can not replace value'
        self.mask = mask

    def __call__(self, data):
        
        data = data.copy()
        tau = data.shape[1]
        t = np.random.randint(0, self.T)
        t0 = np.random.randint(0, tau-t)
        
        if (t0 == t0 + t): return data

        tmax = np.random.randint(t0, t0+t)
        
        if self.mask == 'mean':
            mean = data.mean()
            data[:, t0:tmax, :] = mean
        elif self.mask == 'zero':
            data[:, t0:tmax, :] = 0
            
        return data
