'''create dataset and dataloader'''
import os

import torch.utils.data


def create_dataloader(dataset, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(dataset, batch_size=4, drop_last=False, shuffle=False, num_workers=1, pin_memory=True)
    if phase == 'calibration':
        return torch.utils.data.DataLoader(dataset, batch_size=4, drop_last=False, shuffle=False, num_workers=1, pin_memory=True)
    elif phase == 'validation' or phase == 'val':
        return torch.utils.data.DataLoader(dataset, batch_size=4, drop_last=False, shuffle=False, num_workers=1, pin_memory=True)
    elif phase == 'test':
        return torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)
    else:
        raise NotImplementedError( 'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_root, phase, final_res=128):
    '''create dataset'''
    mode = "LRHR"
    from .LRHR_dataset import LRHRDataset as D
    dataset = D(dataroot=os.path.join(dataset_root,phase),
                datatype='img',
                l_resolution=16,
                r_resolution=128,
                final_res=final_res,
                split=phase,
                data_len=-1,
                need_LR=(mode == 'LRHR'),
                )
    return dataset