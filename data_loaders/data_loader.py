import torchvision.transforms as T

from torch.utils.data import DataLoader
from .market1501 import Market1501
from .bases import ImageDataset

__factory = {
    'market1501': Market1501
}

def make_dataloader(config):
    train_transform = T.compose([
        T.Resize(config.INPUT.TRAIN_SIZE, interpolation=3),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=config.INPUT.PIXEL_MEAN, std=config.INPUT.PIXEL_STD)
    ])

    val_transform = T.compose([
        T.Resize(config.INPUT.TRAIN_SIZE, interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=config.INPUT.PIXEL_MEAN, std=config.INPUT.PIXEL_STD)
    ])

    dataset = __factory[config.DATASETS.NAMES](root = config.DATASETS.ROOT_DIR)

    train_set = ImageDataset(dataset.train, train_transform)
    val_set = ImageDataset(dataset.query, val_transform)

    train_loader = DataLoader(train_set, batch_size=config.SOLVER.IMS_PER_BATCH, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.SOLVER.IMS_PER_BATCH, shuffle=True)

    return train_loader, val_loader

