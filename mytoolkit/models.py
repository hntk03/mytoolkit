import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class L2Constraint(nn.Module):


    def __init__(self, model, alpha):
        super(L2Constraint, self).__init__()

        self.model = model
        self.alpha = alpha


    def forward(self, x):

        x = self.model(x)
        l2 = torch.sqrt(torch.pow(x, 2).sum(dim=1))
        x = self.alpha * (x / l2)
        return x





