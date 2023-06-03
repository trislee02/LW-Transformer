from PIL import Image
from torch.utils.data import Dataset

class BaseImageDataset:
    def __init__(self):
        pass

    def statistic(self):
        pass

class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        ''' Args:
                dataset: A list of tuple which contains (image_path, label)
                transform: A torchvision.transform.Compose object
        '''
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image_path, label = self.dataset[index]

        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)

        return image, label
