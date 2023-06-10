import torch
import os
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

def freeze_all_block(model):
    for block in model.base_model.blocks:
        for param in block.parameters():
            param.requires_grad = False

def unfreeze_blocks(model, num_blocks):
    block_from = model.num_blocks - num_blocks
    if block_from >= 0:
        for block in model.base_model.blocks[block_from:]:
            for param in block.parameters():
                param.requires_grad = True
        return True
    else:
        return False

def do_train(config, model, train_dataloader, val_dataloader, loss_fn, optimizer):
    num_epochs = config.SOLVER.MAX_EPOCHS
    device = config.MODEL.DEVICE
    best_acc = 0.0
    epoch = 0
    num_unfrozen_blocks = 0
    freeze_all_block(model)

    checkpoint = load_checkpoint(config, model, optimizer, device=device)
    if checkpoint is not None:
        model, optimizer, best_acc, last_epoch, last_num_unfrozen_blocks = checkpoint
        unfreeze_blocks(model, last_num_unfrozen_blocks)
        epoch = last_epoch + 1
        num_unfrozen_blocks = last_num_unfrozen_blocks + 1

    while epoch < num_epochs:
        if config.SOLVER.BLOCKWISE_FINETUNE and epoch % config.SOLVER.UNFREEZE_BLOCKS == 0:
            frozen = unfreeze_blocks(model, num_unfrozen_blocks)
            if frozen:
                print(f'\nUnfroze {num_unfrozen_blocks} blocks')
                num_unfrozen_blocks += 1    
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if num_unfrozen_blocks > 0:
            print("\nUnfrozen Blocks: {}, Trainable Params: {}".format(num_unfrozen_blocks - 1, trainable_params))
        else:  print("\nTrainable Params: {}".format(trainable_params))

        print(f"\nEpoch {epoch}: ========================")

        train_one_epoch(model, train_dataloader, loss_fn, optimizer, device=device)

        val_loss, val_acc = validate(model, val_dataloader, loss_fn, device=device)

        save_checkpoint_path = os.path.join(config.OUTPUT_DIR, config.MODEL.NAME + '_checkpoint_epoch_{}_acc_{:.4f}.ckpt'.format(epoch, val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            save_model_path = os.path.join(config.OUTPUT_DIR, config.MODEL.NAME + '_model_epoch_{}_acc_{:.4f}.pth'.format(epoch, best_acc))
            save_checkpoint(model, epoch, optimizer, best_acc, num_unfrozen_blocks-1, save_checkpoint_path, config.MODEL.DEVICE)
            if config.SAVED_MODEL:
                save_model(model, save_model_path, config.MODEL.DEVICE)
            print(f"Saved model at {save_model_path}")
            print(f"Saved checkpoint at {save_checkpoint_path}")

        epoch += 1        