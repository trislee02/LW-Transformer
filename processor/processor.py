import torch
from tqdm import tqdm

def train_one_epoch(model, train_dataloader, loss_fn, optimizer, device='cuda'):
    # Training
    model.train()
    model.to(device)
    epoch_loss = 0.0
    epoch_acc = 0.0

    for data, target in tqdm(train_dataloader):
        data, target = data.to(device), target.to(device)

        # 1. Forward pass
        feature, preds = model(data)

        # 2. Calculate loss
        loss = loss_fn(preds, target)
        epoch_loss += loss

        # 3. Refresh optimizer
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate accuracy
        preds = torch.argmax(preds, dim=1)
        acc = torch.eq(preds, target).sum().item() / len(target)
        epoch_acc += acc

        break

    epoch_loss /= len(train_dataloader)
    epoch_acc /= len(train_dataloader)
    
    print(f"\nTrain loss: {epoch_loss:.5f} - Train acc: {epoch_acc:.5f}")

def do_train(config, model, train_dataloader, val_dataloader, loss_fn, optimizer):
    num_epochs = config.SOLVER.MAX_EPOCHS
    device = config.MODEL.DEVICE
    
    model.to(device)

    for epoch in range(num_epochs):
        train_one_epoch(config, model, train_dataloader, val_dataloader, loss_fn, optimizer, device=device)