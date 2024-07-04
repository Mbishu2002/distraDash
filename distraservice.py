import torch
import Distra from distra

def load_model():
    model = Distra()
    state_dict = torch.load('distra_001.pth')
    model.load_state_dixt(state_dict)
    model.eval()
    return model

def predict_rainfall(model, rainfall_amount):
    # Prepare input data for the model
    input_data = torch.tensor([[rainfall_amount]], dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_data)
    return prediction.item()  
