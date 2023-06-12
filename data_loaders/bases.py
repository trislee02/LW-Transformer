from torch.utils.data import Dataset

class BaseImageDataset:
    def __init__(self):
        pass

    def get_num_classes(self, dataset):
        labels = []
        for path, label in dataset:
            labels += [label]

        labels = set(labels)
        num_classes = len(labels)

        return num_classes

class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, label2id_map={}):
        ''' Args:
                dataset: A list of tuple which contains (image_path, label)
                transform: A torchvision.transform.Compose object
        '''
        self.dataset = dataset
        self.transform = transform
        self.ids = []
        for path, label in dataset:
            self.ids += [label2id_map[label]]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
