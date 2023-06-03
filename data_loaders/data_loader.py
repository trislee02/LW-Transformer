import torchvision.transforms as T

def make_dataloader(config):
    train_transform = T.compose([
        T.Resize(config.INPUT.TRAIN_SIZE, interpolation=3),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=config.INPUT.PIXEL_MEAN, std=config.INPUT.PIXEL_STD)
    ])

    test_transform = T.compose([
        T.Resize(config.INPUT.TRAIN_SIZE, interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=config.INPUT.PIXEL_MEAN, std=config.INPUT.PIXEL_STD)
    ])

    return train_transform, test_transform

