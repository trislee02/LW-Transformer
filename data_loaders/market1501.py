import os.path
import glob
from .bases import BaseImageDataset

class Market1501(BaseImageDataset):

    dataset_dir = 'market1501'
    def __init__(self, root, verbose=True):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = os.path.join(self.dataset_dir, 'query')
        self.gallery_dir = os.path.join(self.dataset_dir, 'bounding_box_test')

        self.train = self.__process_dir(self.train_dir)
        self.query = self.__process_dir(self.query_dir)
        self.gallery = self.__process_dir(self.gallery_dir)

        if verbose:
          print("Train set: {} images".format(len(self.train)))
          print("Query set: {} images".format(len(self.query)))
          print("Gallery set: {} images".format(len(self.gallery)))
          
        self.train_num_classes = self.get_num_classes(self.train)

    def __process_dir(self, dir):
        img_paths = glob.glob(os.path.join(dir, '*.jpg'))
        dataset = []
        for path in img_paths:
            label = int(path.split('/')[-1].split('_')[0])
            dataset.append((path, label))

        return dataset
