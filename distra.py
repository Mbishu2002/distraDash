import torch.nn as nn

class Distra(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Distra, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if x.dim() == 0:
            raise ValueError("Input tensor has no dimensions")

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))

        out_last_step = out[:, -1, :]

        out_fc = self.fc(out_last_step)

        return out_fc

