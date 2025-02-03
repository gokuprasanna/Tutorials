
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

# Load configuration file
with open("config.json", "r") as config_file:
    config = json.load(config_file)

# Recurrent Neural Network (RNN)
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn_layers = nn.ModuleList()
        input_size = config["RNN"]["input_size"]
        for _ in range(config["RNN"]["num_layers"]):
            self.rnn_layers.append(nn.RNN(input_size, config["RNN"]["hidden_size"], batch_first=True))
            input_size = config["RNN"]["hidden_size"]
        self.fc = nn.Linear(config["RNN"]["hidden_size"], config["RNN"]["num_classes"])

    def forward(self, x):
        for rnn in self.rnn_layers:
            x, _ = rnn(x)
        out = self.fc(x[:, -1, :])
        return out

# Long Short-Term Memory (LSTM)
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm_layers = nn.ModuleList()
        input_size = config["LSTM"]["input_size"]
        for _ in range(config["LSTM"]["num_layers"]):
            self.lstm_layers.append(nn.LSTM(input_size, config["LSTM"]["hidden_size"], batch_first=True))
            input_size = config["LSTM"]["hidden_size"]
        self.fc = nn.Linear(config["LSTM"]["hidden_size"], config["LSTM"]["num_classes"])

    def forward(self, x):
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        out = self.fc(x[:, -1, :])
        return out

# Gated Recurrent Unit (GRU)
class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.gru_layers = nn.ModuleList()
        input_size = config["GRU"]["input_size"]
        for _ in range(config["GRU"]["num_layers"]):
            self.gru_layers.append(nn.GRU(input_size, config["GRU"]["hidden_size"], batch_first=True))
            input_size = config["GRU"]["hidden_size"]
        self.fc = nn.Linear(config["GRU"]["hidden_size"], config["GRU"]["num_classes"])

    def forward(self, x):
        for gru in self.gru_layers:
            x, _ = gru(x)
        out = self.fc(x[:, -1, :])
        return out

# Artificial Neural Network (ANN)
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        layers = []
        in_size = config["ANN"]["input_size"]
        for hidden_size in config["ANN"]["hidden_layers"]:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, config["ANN"]["num_classes"]))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_layers(x)

