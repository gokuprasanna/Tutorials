import torch
import torch.nn as nn
import torch.nn.functional as F
import json

# Load configuration file
with open("config.json", "r") as config_file:
    config = json.load(config_file)

# Convolutional Neural Network (CNN)
class CNN(nn.Module):
    def __init__(self, input_channels=config["CNN"]["input_channels"], num_classes=config["CNN"]["num_classes"]):
        super(CNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        
        for out_channels, kernel_size in config["CNN"]["conv_layers"]:
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1))
            in_channels = out_channels
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_channels * 7 * 7, config["CNN"]["fc_units"])
        self.fc2 = nn.Linear(config["CNN"]["fc_units"], num_classes)

    def forward(self, x):
        for conv in self.conv_layers:
            x = self.pool(F.relu(conv(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

