"""Discriminator architectures used as density-ratio estimators for OBRS.

'mlp' is the starter Discriminator from model.py (unchanged).
'cnn' is a small convolutional classifier with the same interface:
input (N, 784) in [-1, 1], output (N, 1) = probability that x is real.
Only the GENERATOR architecture is fixed by the assignment.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Discriminator


class DiscriminatorCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)    # 28 -> 14
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)   # 14 -> 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 1)
        self.drop = nn.Dropout(dropout)

    def logit(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = self.drop(x.flatten(1))
        x = self.drop(F.leaky_relu(self.fc1(x), 0.2))
        return self.fc2(x)

    def forward(self, x):
        return torch.sigmoid(self.logit(x))


def build_D(arch='mlp'):
    if arch == 'mlp':
        return Discriminator(784)
    if arch == 'cnn':
        return DiscriminatorCNN()
    raise ValueError(f'unknown discriminator architecture: {arch}')


def n_params(model):
    return sum(p.numel() for p in model.parameters())
