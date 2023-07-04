class Trainer:
    def train_one_epoch(model, train_dataloader, loss_fn, optimizer, scheduler, device='cuda'):
        raise NotImplementedError

    def validate(model, val_dataloader, loss_fn, device='cuda'):
        raise NotImplementedError