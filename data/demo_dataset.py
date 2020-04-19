import os.path
from data.base_dataset import BaseDataset, get_transform, get_params
from data.data import relabel_dataset
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np

class DemoDataset(BaseDataset):

    def __init__(self,opt):
        BaseDataset.__init__(self,opt)
        if self.opt.isTrain:
            self.dir_no_label_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/unlabeldata/trainA'
            self.dir_no_label_B = os.path.join(opt.dataroot, opt.phase + 'B') # create a path '/path/to/unlabeldata/trainB'
            self.no_label_A_paths = sorted(make_dataset(self.dir_no_label_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
            self.no_label_B_paths = sorted(make_dataset(self.dir_no_label_B, opt.max_dataset_size))
            self.A_size = len(self.no_label_A_paths)  # get the size of dataset A
            self.B_size = len(self.no_label_B_paths)
            self.dir_label = os.path.join(opt.dataroot, opt.phase) # get the image directory
            self.label_paths = sorted(make_dataset(self.dir_label, opt.max_dataset_size))  # get image paths
            self.label_size = len(self.label_paths)
            self.no_label_size = len(self.no_label_A_paths)
            assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image

        else:
            self.test=os.path.join(opt.dataroot,opt.phase)
            self.test_paths = sorted(make_dataset(self.test,opt.max_dataset_size))
            self.test_size = len(self.test_paths)

        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc
        # self.transform = get_transform(self.opt, grayscale=(self.input_nc == 1))
        self.transform_A = get_transform(self.opt, grayscale=(self.input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(self.output_nc == 1))

    def __getitem__(self, index):
        if self.opt.isTrain:
            index_A = random.randint(0, 10000)
            #index = np.random.permutation(np.arange(os.listdir(self.AB_paths)))
            label_path,_ = self.label_paths[int(index_A % self.label_size)]
            label_img = Image.open(label_path).convert('RGB')
            w, h = label_img.size
            w2 = int(w / 2)
            label_A = label_img.crop((0, 0, w2, h))
            label_B = label_img.crop((w2, 0, w, h))
            transform_params = get_params(self.opt, label_A.size)
            label_A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
            label_B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
            label_A = label_A_transform(label_A)
            label_B = label_B_transform(label_B)
            no_label_A,_ = self.no_label_A_paths[int(index_A % self.A_size)]
            index_B = random.randint(0, self.B_size)
            no_label_B = self.no_label_B_paths[int(index_B % self.B_size)]
            A_img = Image.open(no_label_A).convert('RGB')
            B_img = Image.open(no_label_B).convert('RGB')
            # apply image transformation
            A = self.transform_A(A_img)
            B = self.transform_B(B_img)
            return {'label_A': label_A, 'label_B': label_B, 'label_A_paths': label_path, 'label_B_paths': label_path
                    ,'no_label_A': A, 'no_label_A_paths': no_label_A, 'no_label_B': B, 'no_label_B_paths': no_label_B}
        else:
            path = self.test_paths[int(index % self.test_size)]
            img = Image.open(path).convert('RGB')
            A = self.transform_A(img)
            return {'A': A, 'A_paths': path}


    def __len__(self):
        if self.opt.isTrain:
            return self.A_size + self.label_size
        else:
            return  self.test_size



