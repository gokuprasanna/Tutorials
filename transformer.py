# Transformer Model
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=config["Transformer"]["input_dim"], nhead=config["Transformer"]["num_heads"])
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=config["Transformer"]["num_layers"])
        self.fc = nn.Linear(config["Transformer"]["input_dim"], config["Transformer"]["num_classes"])

    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x.mean(dim=1))
        return x

# MAMBA Model (Placeholder, since it's a specialized model)
class MAMBA(nn.Module):
    def __init__(self):
        super(MAMBA, self).__init__()
        self.lstm_layers = nn.ModuleList()
        input_size = config["MAMBA"]["input_size"]
        for _ in range(config["MAMBA"]["num_layers"]):
            self.lstm_layers.append(nn.LSTM(input_size, config["MAMBA"]["hidden_size"], batch_first=True))
            input_size = config["MAMBA"]["hidden_size"]
        self.fc = nn.Linear(config["MAMBA"]["hidden_size"], config["MAMBA"]["num_classes"])

    def forward(self, x):
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        out = self.fc(x[:, -1, :])
        return out
