import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from OrganizedClassifyNetwork import BasicBlock, Bottleneck, ResNet1D
import numpy as np
from scipy import interpolate

def resnet18_1d(num_classes=1000):
    return ResNet1D(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def resnet34_1d(num_classes=1000):
    return ResNet1D(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def resnet50_1d(num_classes=1000):
    return ResNet1D(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

def resnet101_1d(num_classes=1000):
    return ResNet1D(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)

def resnet152_1d(num_classes=1000):
    return ResNet1D(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)

class WarmupCosineLR:
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0.0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.lr_scheduler = CosineAnnealingLR(optimizer, max_epochs - warmup_epochs, eta_min=min_lr)

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            self.lr_scheduler.step(epoch - self.warmup_epochs)
            lr = self.optimizer.param_groups[0]['lr']
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


 # Define custom dataset class
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    

# interpolation data
def interpo(x, y, start, end, points, kind='linear'):
     # Generate new sampling points
    xnew = np.linspace(start, end, points)
    y_update = []
    f = []
    for i in range(len(y)):
        y_update.append([])
         # Create interpolation function
        fi = interpolate.interp1d(x[i], y[i], kind=kind, bounds_error=False, fill_value=0)
        f.append(fi)
         # Calculate interpolation results
        ynew = fi(xnew)
        y_update[i].append(ynew)

    y_update = np.array(y_update)
    y_update = np.squeeze(y_update)
    return y_update
