import torch

def save_checkpoint(model, epoch, optimizer, best_acc, num_unfrozen_blocks, path, device='cpu'):
    model.cpu()
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'num_unfrozen_blocks': num_unfrozen_blocks
        }, path)
    model.to(device)  

def load_checkpoint(config, model, optimizer, device='cpu'):
    checkpoint = torch.load(config.SOLVER.CHECKPOINT_PATH) if config.SOLVER.RESUME_TRAINING else None

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_acc = checkpoint['best_acc']
        last_epoch = checkpoint['epoch']
        last_num_unfrozen_blocks = checkpoint['num_unfrozen_blocks']

        # Move optimizer state to appropriate device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
        return model, optimizer, best_acc, last_epoch, last_num_unfrozen_blocks
    
    return None

def save_model(model, path, device='cpu'):
    model.cpu()
    torch.save(model.state_dict(), path)
    model.to(device)

def load_model(model, path, device='cpu'):
    model.cpu()
    # Load the model state dictionary from the .pth file
    checkpoint = torch.load(path)
    # Load the model weights
    model.load_state_dict(checkpoint, strict=False)
    #
    model.to(device)
    model.eval()