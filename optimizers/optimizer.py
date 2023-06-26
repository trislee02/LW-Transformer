import torch.optim as optim

def make_optimizer(config, model):
    if config.SOLVER.OPTIMIZER_NAME == "ADAM":
        return optim.Adam(model.parameters(), weight_decay=config.SOLVER.WEIGHT_DECAY, lr=config.SOLVER.BASE_LR)
    elif config.SOLVER.OPTIMIZER_NAME == "SGD":
        return optim.SGD(model.parameters(), lr=config.SOLVER.BASE_LR, weight_decay=config.SOLVER.WEIGHT_DECAY, momentum=config.SOLVER.MOMENTUM)