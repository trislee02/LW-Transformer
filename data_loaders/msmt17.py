import os.path
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from .bases import BaseImageDataset


class MSMT17(BaseImageDataset):

    dataset_dir = 'msmt17'
    def __init__(self, root, verbose=True):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, 'mask_train_v2')
        self.test_dir = os.path.join(self.dataset_dir, 'mask_test_v2')

        self.train, self.train_labels, self.train_num_classes = self.__process_dir(self.dataset_dir, self.train_dir, 'list_train.txt')
        self.val, self.val_labels, self.val_num_classes = self.__process_dir(self.dataset_dir, self.train_dir, 'list_val.txt')
        self.query, self.query_labels, self.query_num_classes = self.__process_dir(self.dataset_dir, self.test_dir, 'list_query.txt')
        self.gallery, self.gallery_labels, self.gallery_num_classes = self.__process_dir(self.dataset_dir, self.test_dir, 'list_gallery.txt')

        if verbose:
          print("Train set: {} images".format(len(self.train)))
          print("Val set: {} images".format(len(self.val)))
          print("Query set: {} images".format(len(self.query)))
          print("Gallery set: {} images".format(len(self.gallery)))
          

    def __process_dir(self, dataset_dir, image_dir, split_file):
        label_ids = {} # { label: id }

        split_file = os.path.join(dataset_dir, split_file)

        dataset = []
        with open(split_file) as f:
            lines = f.readlines()
            for line in lines:
                path = os.path.join(image_dir, line.strip().split(' ')[0])
                id = int(line.strip().split(' ')[1])
                dataset.append((path, id))
                label_ids[id] = str(id)

        return dataset, label_ids, len(label_ids)
