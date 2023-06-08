import torch

def save_network(model, path, device='cpu'):
    model.cpu()
    torch.save(model, path)
    model.to(device)    