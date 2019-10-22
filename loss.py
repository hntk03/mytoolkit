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
        batch_size = output.size()[0]
        delta = torch.eye(self.num_classes)[target]
        q = (1- self.epsilon) * delta + self.epsilon * self.u_k
        q = q.to(device)
        
        return F.kl_div(output, q, reduction='batchmean')
