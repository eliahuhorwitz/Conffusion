import random
from io import BytesIO

import data.util as Util
import lmdb
import torch
from PIL import Image
from torch.utils.data import Dataset


class BoundsDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='finetune', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        self.lower_bounds = Util.make_dataset_extracted_bounds(f"{dataroot}/sampled_bounds/lower_bounds/")
        self.upper_bounds = Util.make_dataset_extracted_bounds(f"{dataroot}/sampled_bounds/upper_bounds/")

        self.sr_path = Util.get_paths_from_images('{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
        self.hr_path = Util.get_paths_from_images('{}/hr_{}'.format(dataroot, r_resolution))
        if self.need_LR:
            self.lr_path = Util.get_paths_from_images('{}/lr_{}'.format(dataroot, l_resolution))

        if data_len > 0:
            self.sr_path = self.sr_path[:int(data_len)]
            self.hr_path = self.hr_path[:int(data_len)]
            self.lower_bounds = self.lower_bounds [:int(data_len)]
            self.upper_bounds = self.upper_bounds[:int(data_len)]
            if self.need_LR:
                self.lr_path = self.lr_path[:int(data_len)]



        self.dataset_len = len(self.hr_path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = Image.open(self.hr_path[index]).convert("RGB")
        img_SR = Image.open(self.sr_path[index]).convert("RGB")
        img_LR = Image.open(self.lr_path[index]).convert("RGB")

        [img_LR, img_SR, img_HR] = Util.transform_augment([img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1))

        lower_bound = torch.load(self.lower_bounds[index])
        upper_bound = torch.load(self.upper_bounds[index])
        return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index, "lower_bound": lower_bound, "upper_bound": upper_bound}

