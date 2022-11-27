import argparse
import logging
import os

import core.logger as Logger
import data as Data
import model as Model
import torch
import tqdm
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/extract_bounds_16_128.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['calibration', 'validation', 'test'], default='calibration')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('--enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    parser.add_argument('--n_soch_samples', type=int, default=200)
    parser.add_argument('--quantile', type=float, default=0.95)
    parser.add_argument('--finetune_loss', type=str, default=None)
    parser.add_argument('--distributed_worker_id', type=int, required=True)
    parser.add_argument('--prediction_time_step', type=int, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--lr', type=float, default=None)

    
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])


    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == args.phase:
            phase_dataset = Data.create_dataset(dataset_opt, phase)
            phase_loader = Data.create_dataloader(phase_dataset, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    logger.info('Begin Bounds Extraction.')
    sample_idx = 0

    bounds_path = f'{opt["datasets"][args.phase]["dataroot"]}/sampled_bounds'
    os.makedirs(bounds_path, exist_ok=True)

    calibration_set_size = opt['datasets'][args.phase]['data_len']
    sr_res = opt["datasets"][args.phase]["r_resolution"]
    n_img_channels = 3
    n_soch_samples = args.n_soch_samples

    upper_quantile = args.quantile
    lower_quantile = (1-args.quantile)

    with torch.no_grad():
        for _, phase_data in enumerate(phase_loader):
            sample_idx += 1
            sample_variations = torch.zeros((n_soch_samples, n_img_channels, sr_res, sr_res))

            for soch_idx in tqdm.tqdm(range(n_soch_samples), desc=f'Stochastic sample for batch {sample_idx-1}', total=n_soch_samples):
                logger.info(f'Sample {sample_idx}, variation {soch_idx}...')
                diffusion.feed_data(phase_data)
                diffusion.test(continous=True)
                visuals = diffusion.get_current_visuals(need_LR=False)
                sample_variations[soch_idx] = visuals['SR'][-1]

            sample_upper_bound = torch.quantile(sample_variations, upper_quantile, dim=0)
            sample_lower_bound = torch.quantile(sample_variations, lower_quantile, dim=0)


            filename_suffix = f'_{n_soch_samples}soch_smpls'
            sample_file_name = ".".join(phase_data['Path'][0].split(".")[0:-1])  # ALG This is in order to trim the file extension but handle filenames with multiple .'s

            os.makedirs(f'{bounds_path}/upper_bounds/', exist_ok=True)
            os.makedirs(f'{bounds_path}/lower_bounds/', exist_ok=True)
            torch.save(sample_upper_bound, f'{bounds_path}/upper_bounds/{sample_file_name}_sampled_upper_bounds{filename_suffix}.pt')
            torch.save(sample_lower_bound, f'{bounds_path}/lower_bounds/{sample_file_name}_sampled_lower_bounds{filename_suffix}.pt')



