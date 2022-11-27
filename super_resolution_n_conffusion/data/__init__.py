'''create dataset and dataloader'''
import logging
from re import split

import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'calibration':
        return torch.utils.data.DataLoader(dataset, batch_size=dataset_opt['batch_size'], drop_last=False, shuffle=dataset_opt['use_shuffle'], num_workers=dataset_opt['num_workers'], pin_memory=True)
    elif phase == 'validation' or phase == 'val':
        return torch.utils.data.DataLoader(dataset, batch_size=dataset_opt['batch_size'], drop_last=False, shuffle=False, num_workers=dataset_opt['num_workers'], pin_memory=True)
    elif phase == 'test':
        return torch.utils.data.DataLoader(dataset, batch_size=dataset_opt['batch_size'], shuffle=False, drop_last=False, num_workers=dataset_opt['num_workers'], pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    from data.LRHR_dataset import LRHRDataset as D
    if dataset_opt['skip_n_samples'] is None:
        dataset_opt['skip_n_samples'] = -1
    dataset = D(dataroot=dataset_opt['dataroot'],
                l_resolution=dataset_opt['l_resolution'],
                r_resolution=dataset_opt['r_resolution'],
                split=phase,
                data_len=dataset_opt['data_len'],
                need_LR=(mode == 'LRHR'),
                skip_n_samples=dataset_opt['skip_n_samples']
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset



def create_dataset_for_bounds_finetuning(dataset_opt, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    from data.bounds_finetune_dataset import BoundsDataset as D
    dataset = D(dataroot=dataset_opt['dataroot'],
                datatype=dataset_opt['datatype'],
                l_resolution=dataset_opt['l_resolution'],
                r_resolution=dataset_opt['r_resolution'],
                split=phase,
                data_len=dataset_opt['data_len'],
                need_LR=(mode == 'LRHR'),
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset