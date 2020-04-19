"""This package includes all the modules related to data loading and preprocessing
 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import os
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
from torch.utils.data.sampler import Sampler,BatchSampler, SubsetRandomSampler
import itertools
import random

import torchvision.datasets
from util.util import *

NO_LABEL = -1
def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".
    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"#data.demo_dataset
    #data.unaligned_dataset
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'#demodataset
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options



def relabel_dataset(dataset, labels):
     unlabeled_idxs = []
     for idx in range(len(dataset.label_paths)):
         path = dataset.label_paths[idx]
         filename = os.path.basename(path)
         if filename in labels:
             label_idx = 'label'
             dataset.label_paths[idx] = path, label_idx
             del labels[filename]
     for idx in range(len(dataset.no_label_A_paths)):
         path = dataset.no_label_A_paths[idx]
         dataset.no_label_A_paths[idx] = path, NO_LABEL
         unlabeled_idxs.append(idx)
     if len(labels) != 0:
         message = "List of unlabeled contains {} unknown files: {}, ..."
         some_missing = ', '.join(list(labels.keys())[:5])
         raise LookupError(message.format(len(labels), some_missing))

     labeled_idxs = sorted(set(range(len(dataset.label_paths))))

     return labeled_idxs, unlabeled_idxs

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    迭代两组索引

     'epoch'是通过主要指数的一次迭代。 在该epoch期间，次要指数根据需要重复多次
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

def create_dataset(opt):
    """Create a dataset given the option.
    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'
    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""



    def __init__(self, opt):
        """Initialize this class
        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """

        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        assert_exactly_one([opt.exclude_unlabeled, opt.labeled_batch_size])
        print("dataset [%s] was created" % type(self.dataset).__name__)
        if opt.isTrain:
            if opt.labels:
                with open(opt.labels) as f:
                    labels = dict(line.split(' ') for line in f.read().splitlines())
                opt.labeled_idxs, opt.unlabeled_idxs = relabel_dataset(self.dataset, labels)
            if opt.exclude_unlabeled:
                sampler = SubsetRandomSampler(opt.labeled_idxs)
                opt.batch_sampler = BatchSampler(sampler, opt.batch_size, drop_last=True)
            elif opt.labeled_batch_size:
                opt.batch_sampler = TwoStreamBatchSampler(opt.unlabeled_idxs, opt.labeled_idxs, opt.batch_size,
                                                           opt.labeled_batch_size)#!！!！!!！!！!！!！
            else:
                assert False, "labeled batch size {}".format(opt.labeled_batch_size)

#        self.dataloader = torch.utils.data.DataLoader(
#                                        self.dataset,
#                                        batch_sampler = batch_sampler,
#                                        num_workers = int(opt.num_threads),
#                                        pin_memory = True)
        if opt.isTrain:
            self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                    batch_sampler=opt.batch_sampler,
                                    num_workers=int(opt.num_threads),
                                    pin_memory=True)
        else:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=1,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.num_threads))


    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
#            print(data)

