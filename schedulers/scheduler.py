from torch.optim.lr_scheduler import StepLR

def make_scheduler(config, optimizer):
    return StepLR(optimizer, step_size=config.SOLVER.STEP_LR_SIZE, gamma=config.SOLVER.GAMMA)