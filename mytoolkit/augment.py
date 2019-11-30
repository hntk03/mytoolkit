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
        upsilon = data.shape[0]
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

class RandomErasing(object):

    '''
    https://arxiv.org/pdf/1708.04896.pdf
    '''
   
    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=3.3, mask_mode='zero'):

        self.p = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2

        assert mask_mode == 'zero' or mask_mode == 'random'
        self.mask_mode = mask_mode

    def __call__(self, data):
        data = data.copy()

        if self.p >= np.random.random():
            return data


        H, W, C = data.shape
        S = H * W

        while True:
            Se = np.random.uniform(self.sl, self.sh) * S
            re = np.random.uniform(self.r1, self.r2)
            

            He = int(np.sqrt(Se*re))
            We = int(np.sqrt(Se/re));

            xe = np.random.randint(0, W)
            ye = np.random.randint(0, H)

            if((xe + We <= W) and (ye + He) <= H):
                break
        if self.mask_mode == 'zero' :
            mask = np.zeros((He, We, C), dtype='float32')
        elif self.mask_mode == 'random':
            mask = np.random.randint(0, 255, (He, We, C))

        data[ye:ye + He, xe:xe + We, :] = mask

        return data



