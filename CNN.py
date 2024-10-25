from modulefinder import Module
"""
CNN
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import functional as F

class cnn(nn.Module):
    def __init__(self, n_layers, kernel_sizes, strides, paddings, output_paddings):
        super(cnn, self).__init__()
        self.n_layers = n_layers
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.output_paddings = output_paddings

    def network(self):
        """
        build a network with CNNs 
        """



    def forward(self, x):
        """
        forward computation for convolutional network
        """



