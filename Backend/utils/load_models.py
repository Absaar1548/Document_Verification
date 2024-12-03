from Models.Verfication.src.models.cnn import MODELS
import torch

def load_verfication_model(model_name: str = 'custom', model_weights: str = 'Model/models/custom_best.pth', device: str = 'cpu'):
    print(f"Loading model '{model_name}' with weights from '{model_weights}'")
    
    # Load the model
    model = MODELS[model_name]
    model.load_state_dict(torch.load(model_weights, map_location=device, weights_only=True))  # Map weights to the correct device
    model.to(device)  # Move model to the specified device
    model.eval()  # Set the model to evaluation mode
    
    print(f"Model '{model_name}' loaded successfully and moved to '{device}'")
    return model

def load_cleaning_model():
    return None