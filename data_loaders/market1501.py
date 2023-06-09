import os.path
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from .bases import BaseImageDataset


class Market1501(BaseImageDataset):

    dataset_dir = 'market1501'
    def __init__(self, root, verbose=True):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = os.path.join(self.dataset_dir, 'query')
        self.gallery_dir = os.path.join(self.dataset_dir, 'bounding_box_test')

        self.train, self.train_labels, self.train_num_classes = self.__process_dir(self.train_dir)
        self.query, self.query_labels, self.query_num_classes = self.__process_dir(self.query_dir)
        self.gallery, self.gallery_labels, self.gallery_num_classes = self.__process_dir(self.gallery_dir)

        X = np.array([data[0] for data in self.train])
        y = np.array([data[1] for data in self.train])

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y)

        self.train = [(X_train[i], y_train[i]) for i in range(len(X_train))]
        self.val = [(X_val[i], y_val[i]) for i in range(len(X_val))]

        if verbose:
          print("Train set: {} images".format(len(self.train)))
          print("Val set: {} images".format(len(self.val)))
          print("Query set: {} images".format(len(self.query)))
          print("Gallery set: {} images".format(len(self.gallery)))
          

    def __process_dir(self, dir):
        img_paths = glob.glob(os.path.join(dir, '*.jpg'))
        label_ids = {} # { label: id }
        id_labels = {} # { id: label }
        max_class = 0

        dataset = []
        for path in img_paths:
            id = str(path.split('/')[-1].split('_')[0])
            if id not in id_labels:
                id_labels[id] = max_class
                label_ids[max_class] = id
                max_class += 1

            # image = Image.open(path)

            dataset.append((path, id_labels[id]))

        return dataset, label_ids, max_class
