import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class LSRLoss(nn.Module):
    '''
    Label Smoothing
    https://arxiv.org/pdf/1512.00567.pdf
    '''
    
    def __init__(self, epsilon, num_classes):
        assert 0.0 < epsilon <= 1.0
        super(LSRLoss, self).__init__()
        
        self.epsilon = epsilon
        self.u_k = 1 / num_classes
        self.num_classes = num_classes
    
    def forward(self, output, target):
        
        device = output.device.type + ':' +str(output.device.index)
        output = F.log_softmax(output, dim=1)
        delta = torch.eye(self.num_classes)[target]
        q = (1- self.epsilon) * delta + self.epsilon * self.u_k
        q = q.to(device)
        
        return F.kl_div(output, q, reduction='batchmean')

class NCrossEntropy(nn.Module):
    '''
        Cross Entropy Loss
        output : (batchsize x K) by predicter
        target : supervised One-hot label 
    '''
    
    def __init__(self):
        super(NCrossEntropy, self).__init__()
    
    def forward(self, output, target):
        return - (F.log_softmax(output, dim=1) * target).sum(dim=1).mean()
