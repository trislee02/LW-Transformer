import torch.optim as optim

def make_optimizer(config, model):
    return optim.Adam(model.parameters(), weight_decay=config.SOLVER.WEIGHT_DECAY, lr=config.SOLVER.BASE_LR)