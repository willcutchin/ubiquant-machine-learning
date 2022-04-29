import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_dem):
        super(GRU, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers)

        self.fc = nn.Linear(hidden_size, output_dem)

    def forward(self, x):
        hidden0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, _ = self.gru(x, hidden0.detach())
        out = out[:, -1, :]
        out = self.fc(out)
        return out


