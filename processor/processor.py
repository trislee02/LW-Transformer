import torch
import os
from tqdm import tqdm
from ..utils.network import save_network

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

    epoch_loss /= len(train_dataloader)
    epoch_acc /= len(train_dataloader)
    
    print(f"\nTrain loss: {epoch_loss:.5f} - Train acc: {epoch_acc:.5f}")

def validate(model, val_dataloader, loss_fn, device='cuda'):
    model.eval()
    model.to(device)
    val_loss = 0.0
    val_acc = 0.0

    with torch.inference_mode():
        for data, target in tqdm(val_dataloader):
            data, target = data.to(device), target.to(device)

            # 1. Forward pass
            feature, preds = model(data)

            # 2. Calculate loss
            loss = loss_fn(preds, target)
            val_loss += loss

            # 3. Calculate accuracy
            preds = torch.argmax(preds, dim=1)
            acc = torch.eq(preds, target).sum().item() / len(target)
            val_acc += acc

        val_loss /= len(val_dataloader)
        val_acc /= len(val_dataloader)

    print(f"\nVal loss: {val_loss:.5f} - Val acc: {val_acc:.5f}")

    return val_loss, val_acc

def do_train(config, model, train_dataloader, val_dataloader, loss_fn, optimizer):
    num_epochs = config.SOLVER.MAX_EPOCHS
    device = config.MODEL.DEVICE
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}: ========================")

        # train_one_epoch(model, train_dataloader, loss_fn, optimizer, device=device)

        val_loss, val_acc = validate(model, val_dataloader, loss_fn, device=device)

        if val_acc > best_acc:
            save_path = os.path.join(config.OUTPUT_DIR, config.MODEL.NAME + '_{}.pth'.format(epoch))
            save_network(model, save_path, config.MODEL.DEVICE)

        