import torch
import numpy as np

from trainer_base import Trainer
from tqdm import tqdm


class TripletLossTrainer(Trainer):
    def train_one_epoch(model, train_dataloader, loss_fn, optimizer, scheduler, device='cuda'):
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

        # Update learning rate
        scheduler.step()

        epoch_loss /= len(train_dataloader)
        epoch_acc /= len(train_dataloader)
        
        print(f"\nTrain loss: {epoch_loss:.5f} - Train acc: {epoch_acc:.5f}")

        return epoch_loss, epoch_acc

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