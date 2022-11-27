import random
from io import BytesIO

import sr_dataset.util as Util
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as trans_fn


class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, final_res=128, split='train', data_len=-1, need_LR=False, skip_n_samples=-1):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.final_res = final_res
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        self.skip_n_samples = skip_n_samples

        self.sr_path = Util.get_paths_from_images('{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
        self.hr_path = Util.get_paths_from_images('{}/hr_{}'.format(dataroot, r_resolution))
        if self.need_LR:
            self.lr_path = Util.get_paths_from_images('{}/lr_{}'.format(dataroot, l_resolution))

        if skip_n_samples > 0:
            self.sr_path = self.sr_path[skip_n_samples:]
            self.hr_path = self.hr_path[skip_n_samples:]
            if self.need_LR:
                self.lr_path = self.lr_path[skip_n_samples:]
        if data_len > 0:
            self.sr_path = self.sr_path[:int(data_len)]
            self.hr_path = self.hr_path[:int(data_len)]
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
        img_HR = None
        img_LR = None

        img_HR = Image.open(self.hr_path[index]).convert("RGB")
        img_SR = Image.open(self.sr_path[index]).convert("RGB")

        img_SR_224 = trans_fn.resize(img_SR, 224, trans_fn.InterpolationMode.BICUBIC)
        if self.final_res !=self.r_res:
            img_HR = trans_fn.resize(img_HR, self.final_res, trans_fn.InterpolationMode.BICUBIC)
            img_SR = trans_fn.resize(img_SR, self.final_res, trans_fn.InterpolationMode.BICUBIC)

        if self.need_LR:
            img_LR = Image.open(self.lr_path[index]).convert("RGB")

        if self.need_LR:
            [img_LR, img_SR, img_HR, img_SR_224] = Util.transform_augment([img_LR, img_SR, img_HR, img_SR_224], split=self.split, min_max=(-1, 1))

            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR,'SR_224': img_SR_224, 'Index': index, 'Path' : self.hr_path[index].rsplit("/")[-1].rsplit("\\")[-1]}
        else:
            [img_SR, img_HR] = Util.transform_augment([img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'HR': img_HR, 'SR': img_SR, 'SR_224': img_SR_224, 'Index': index, 'Path' : self.hr_path[index].rsplit("/")[-1].rsplit("\\")[-1]}
