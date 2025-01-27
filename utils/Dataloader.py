import os
from PIL import Image
import numpy as np
import random
import torch
from torchvision import transforms
from torch.utils import data
from torch.utils.data import Dataset


class MBRSDataset(Dataset):

    def __init__(self, path, H=256, W=256, transform_type=None, data_file_layout=None):
        super(MBRSDataset, self).__init__()
        self.H = H
        self.W = W
        self.path = path
        self.transform_type = transform_type
        self.data_file_layout = data_file_layout
        print(f"MBRSDataset::__init__()...")
        print(f"  H, W: {self.H}, {self.W}")
        print(f"  path: {self.path}")
        print(f"  data_file_layout: {self.data_file_layout}")
        print(f"  transform_type  : {self.transform_type}")
        # sub_path_list = self.get_file_list_from_meta_files(self.path)
        if data_file_layout == 'sub_dirs':
            file_name_list = self.get_file_list_literally_from_sub_dir()
        elif data_file_layout is None:
            file_name_list = os.listdir(self.path)
            file_name_list = [f for f in file_name_list if f.endswith('.png')]
        else:
            raise ValueError(f"Invalid data_file_layout: {data_file_layout}")
        file_name_list.sort()
        print(f"  total file cnt: {len(file_name_list)}")
        self.list = file_name_list
        if self.transform_type  == 2:
            tf_cnt = 100
            arr = []
            for p in np.linspace(0.7, 1.4, tf_cnt, endpoint=True):
                tf = transforms.Compose([
                        transforms.RandomCrop((int(H*p), int(W*p))),
                        transforms.Resize((H, W)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ])
                arr.append(tf)
            self.transform_arr = arr
            self.transform_arr_len = len(arr)
            print(f"  transform_arr_len: {self.transform_arr_len}")
        elif self.transform_type == 1:
            self.transform = transforms.Compose([
                        transforms.RandomCrop((H, W)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        elif self.transform_type is None:
            self.transform = transforms.Compose([
                transforms.Resize((int(self.H * 1.1), int(self.W * 1.1))),
                transforms.RandomCrop((self.H, self.W)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            raise ValueError(f"Invalid transform_type: {self.transform_type}")
        print(f"MBRSDataset::__init__()...Done")

    def get_file_list_from_meta_files(self, path):
        train_meta_file, val_meta_file = "train_meta.txt", "val_meta.txt"
        arr1 = self.get_file_list_from_meta_file(self.path, train_meta_file)
        arr2 = self.get_file_list_from_meta_file(self.path, val_meta_file)
        print(f"  {train_meta_file:14s}: {len(arr1)}")
        print(f"  {val_meta_file:14s}: {len(arr2)}")
        sub_path_list = arr1 + arr2
        sub_path_list.sort()
        return sub_path_list

    @staticmethod
    def get_file_list_from_meta_file(path, meta_file_name):
        f_path = os.path.join(path, meta_file_name)
        with open(f_path, 'r') as f:
            line_arr = f.readlines()
        arr = []
        for line in line_arr:
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue
            arr.append(line)
        return arr

    def get_file_list_literally_from_sub_dir(self):
        sub_dir_arr = [f"pdfa-eng-train-{i:04d}" for i in range(0, 11)]
        print(f"  sub_path cnt: {len(sub_dir_arr)}")
        sub_path_list = []
        for sub_dir in sub_dir_arr:
            full_dir = os.path.join(self.path, sub_dir)
            file_name_list = os.listdir(full_dir)
            print(f"  {sub_dir}: {len(file_name_list)}")
            for file_name in file_name_list:
                sub_path = os.path.join(sub_dir, file_name)
                sub_path_list.append(sub_path)
            # for
        # for
        return sub_path_list

    def transform_image(self, image):
        # ignore
        if image.size[0] < self.W / 2 and image.size[1] < self.H / 2:
            return None
        if image.size[0] < image.size[1] / 2 or image.size[1] < image.size[0] / 2:
            return None

        # Augment, ToTensor and Normalize
        image = self.transform(image)

        return image

    def __getitem__(self, index):
        if self.transform_type == 2:
            transform_image = self.transform_arr[index % self.transform_arr_len]
        else:
            transform_image = self.transform
        image = Image.open(os.path.join(self.path, self.list[index])).convert("RGB")
        image = transform_image(image)
        return image

    def __len__(self):
        return len(self.list)
