import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_dem, prob_drop):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=prob_drop)
        self.fc = nn.Linear(hidden_size, output_dem)

    def forward(self, x):
        hidden0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        # print(hidden0)
        out, hidden0 = self.rnn(x, hidden0.detach())
        # print(out)
        out = out[:, -1, :]
        # print(out)
        out = self.fc(out)
        return out

