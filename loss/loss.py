from torch import nn

def make_loss(config):
    return nn.CrossEntropyLoss()

