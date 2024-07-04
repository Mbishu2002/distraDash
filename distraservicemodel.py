import torch
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

input_size = 13
hidden_size = 64
num_layers = 2
output_size = 1

def load_model():
    model = Distra(input_size, hidden_size, num_layers, output_size)
    state_dict = torch.load('distra_001.pth')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_rainfall(model, rainfall_amount):
    # Prepare input data for the model
    input_data = torch.tensor([[rainfall_amount]], dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        prediction = model(input_data)
        lable = prediction.int()
        if lable.item() == 1:
            flood_state = "potential flood"
        elif lable.item() == 0:
            flood_state = "no flood"
        else:
            flood_state = "unknown"

    return prediction.item(), flood_state
