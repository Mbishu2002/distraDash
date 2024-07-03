import torch

def load_model():
    # Replace 'model.pth' with your actual model file
    model = torch.load('distra_001.pth')
    model.eval() 
    return model

def predict_rainfall(model, rainfall_amount):
    # Prepare input data for the model
    input_data = torch.tensor([[rainfall_amount]], dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_data)
    return prediction.item()  