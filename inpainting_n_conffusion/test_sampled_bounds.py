import argparse
import os
from functools import partial

import core.praser as Praser
import core.util as Util
import torch
from core import vis_util
from core.calibration_masked import calibrate_model
from core.eval_utils_masked import get_rcps_metrics
from core.logger import InfoLogger, VisualWriter
from core.praser import init_obj
from core.wandb_logger import WandbLogger
from data import define_dataloader
from torch.utils.data import DataLoader


def run_validation(device, val_loader):
    with torch.no_grad():
        # region INIT VARS
        val_batch_size = val_loader.batch_size
        dataset_len = len(val_loader.dataset)

        res = 256
        n_img_channels = 3
        all_val_sampled_lower_bounds = torch.zeros((min(dataset_len, len(val_loader) * val_batch_size), n_img_channels, res, res), device=device)
        all_val_sampled_upper_bounds = torch.zeros((min(dataset_len, len(val_loader) * val_batch_size), n_img_channels, res, res), device=device)
        all_val_gt_samples = torch.zeros((min(dataset_len, len(val_loader) * val_batch_size), n_img_channels, res, res), device=device)
        all_val_maskes = torch.zeros((min(dataset_len, len(val_loader) * val_batch_size), n_img_channels, res, res), device=device)
        # endregion

        for val_idx, val_data in enumerate(val_loader):
            val_sampled_l_bound = val_data["lower_bound"].to(device)
            val_sampled_u_bound = val_data["upper_bound"].to(device)
            val_sampled_mask = val_data["sampled_masks"].to(device)
            val_gt_image = val_data["gt_image"].to(device)
            val_sampled_l_bound = (torch.zeros_like(val_sampled_l_bound) * (1. - val_sampled_mask) + val_sampled_mask * val_sampled_l_bound)
            val_sampled_u_bound = (torch.zeros_like(val_sampled_u_bound) * (1. - val_sampled_mask) + val_sampled_mask * val_sampled_u_bound)

            # region HOLD ON TO THE SAMPLE
            start_idx = val_idx * val_batch_size
            end_idx = min(dataset_len, ((val_idx + 1) * val_batch_size))
            all_val_sampled_lower_bounds[start_idx:end_idx] = val_sampled_l_bound
            all_val_sampled_upper_bounds[start_idx:end_idx] = val_sampled_u_bound
            all_val_gt_samples[start_idx:end_idx] = val_gt_image
            all_val_maskes[start_idx:end_idx] = val_data['sampled_masks']
            #endregion

        # region NORM TO [0,1]
        # Transform all tensors to [0,1] for calibration and metric calculation
        all_val_sampled_lower_bounds = Util.normalize_tensor(all_val_sampled_lower_bounds)
        all_val_sampled_upper_bounds = Util.normalize_tensor(all_val_sampled_upper_bounds)
        all_val_gt_samples = Util.normalize_tensor(all_val_gt_samples)
        # endregion



        sampled_val_lambda_hat, _, _ = calibrate_model(all_val_sampled_lower_bounds, all_val_sampled_upper_bounds, all_val_gt_samples, all_val_maskes)

        return sampled_val_lambda_hat





def run_test(wandb_logger, device, test_loader, sampled_val_lambda_hat):
    with torch.no_grad():
        test_batch_size = test_loader.batch_size
        dataset_len = len(test_loader.dataset)


        res = 256
        n_img_channels = 3
        all_test_sampled_lower_bounds = torch.zeros((min(dataset_len, len(test_loader) * test_batch_size), n_img_channels, res, res), device=device)
        all_test_sampled_upper_bounds = torch.zeros((min(dataset_len, len(test_loader) * test_batch_size), n_img_channels, res, res), device=device)
        all_test_gt_samples = torch.zeros((min(dataset_len, len(test_loader) * test_batch_size), n_img_channels, res, res), device=device)
        all_test_maskes = torch.zeros((min(dataset_len, len(test_loader) * test_batch_size), n_img_channels, res, res), device=device)
        all_test_partial_gt_images = torch.zeros((min(dataset_len, len(test_loader) * test_batch_size), n_img_channels, res, res))
        all_test_masked_samples = torch.zeros((min(dataset_len, len(test_loader) * test_batch_size), n_img_channels, res, res))

        for test_idx, test_data in enumerate(test_loader):
            test_sampled_l_bound = test_data["lower_bound"].to(device)
            test_sampled_u_bound = test_data["upper_bound"].to(device)
            test_sampled_mask = test_data["sampled_masks"].to(device)
            test_gt_image = test_data["gt_image"].to(device)

            test_sampled_l_bound = (torch.zeros_like(test_sampled_l_bound) * (1. - test_sampled_mask) + test_sampled_mask * test_sampled_l_bound)
            test_sampled_u_bound = (torch.zeros_like(test_sampled_u_bound) * (1. - test_sampled_mask) + test_sampled_mask * test_sampled_u_bound)
            partial_gt = (torch.zeros_like(test_sampled_u_bound) * (1. - test_sampled_mask) + test_sampled_mask * test_gt_image)

            # region HOLD ON TO THE SAMPLE
            start_idx = test_idx * test_batch_size
            end_idx = min(dataset_len, ((test_idx + 1) * test_batch_size))
            all_test_sampled_lower_bounds[start_idx:end_idx] = test_sampled_l_bound
            all_test_sampled_upper_bounds[start_idx:end_idx] = test_sampled_u_bound
            all_test_gt_samples[start_idx:end_idx] = test_gt_image
            all_test_partial_gt_images[start_idx:end_idx] = partial_gt
            all_test_maskes[start_idx:end_idx] = test_data['sampled_masks']
            all_test_masked_samples[start_idx:end_idx] = test_data['masked_samples']
            # endregion

        # region NORM TO [0,1]
        # Transform all tensors to [0,1] for calibration and metric calculation
        all_test_sampled_lower_bounds = Util.normalize_tensor(all_test_sampled_lower_bounds)
        all_test_sampled_upper_bounds = Util.normalize_tensor(all_test_sampled_upper_bounds)
        all_test_gt_samples = Util.normalize_tensor(all_test_gt_samples)
        all_test_partial_gt_images = Util.normalize_tensor(all_test_partial_gt_images)
        all_test_masked_samples = Util.normalize_tensor(all_test_masked_samples)
        # endregion

        # region MULTIPLY BOUNDS BY LAMBDA
        sampled_test_calibrated_l = (all_test_sampled_lower_bounds / sampled_val_lambda_hat).clamp(0., 1.)
        sampled_test_calibrated_u = (all_test_sampled_upper_bounds * sampled_val_lambda_hat).clamp(0., 1.)
        # endregion


        calibrated_sampled_risks_losses, calibrated_sampled_sizes_mean, calibrated_sampled_sizes_median, calibrated_sampled_stratified_risks = get_rcps_metrics(sampled_test_calibrated_l, sampled_test_calibrated_u, all_test_gt_samples, all_test_maskes)

        if wandb_logger:
            image_grid = vis_util.create_image_grid(sampled_test_calibrated_l.detach().cpu() * 2 - 1,
                                           sampled_test_calibrated_u.detach().cpu() * 2 - 1,
                                           all_test_partial_gt_images.detach().cpu() * 2 - 1,
                                           all_test_masked_samples.detach().cpu() * 2 - 1,
                                           all_test_gt_samples.detach().cpu() * 2 - 1)

            wandb_logger.log_image("Test/Images", image_grid, caption="Sampled L, Sampled U, In LR, GT HR", commit=False)

            wandb_logger.log_metrics({'Test/Calibrated Sampled Risk': calibrated_sampled_risks_losses}, commit=False)
            wandb_logger.log_metrics({'Test/Calibrated Sampled Size Mean': calibrated_sampled_sizes_mean}, commit=False)
            wandb_logger.log_metrics({'Test/Calibrated Sampled Size Median': calibrated_sampled_sizes_median}, commit=False)
            calibrated_sampled_stratified_risks_data = dict([[label, test] for (label, test) in
                                                             zip([
                                                                     "Test/Calibrated Sampled Stratified Risks - Short",
                                                                     "Test/Calibrated Sampled Stratified Risks - Short-Medium",
                                                                     "Test/Calibrated Sampled Stratified Risks - Medium-Long",
                                                                     "Test/Calibrated Sampled Stratified Risks - Long"],
                                                                 calibrated_sampled_stratified_risks.detach().cpu().numpy())])
            wandb_logger.log_metrics(calibrated_sampled_stratified_risks_data)




def main_worker(gpu, opt):

    # region INIT SEEDS
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu
    torch.backends.cudnn.enabled = True
    Util.set_seed(opt['seed'])

    device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
    # endregion

    # region INIT WANDB
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('epoch')
        wandb.define_metric("test/Calibrated Sampled Size Mean", summary="min")
        wandb.define_metric("test/Calibrated Sampled Size Median", summary="min")
        wandb.define_metric("Calibrated Sampled Size Mean", summary="min")
        wandb.define_metric("Calibrated Sampled Size Median", summary="min")
    else:
        wandb_logger = None
    # endregion

    # region INIT LOGGER
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)  
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))
    # endregion

    # region INIT DATASETS
    phase_loader, val_loader = define_dataloader(phase_logger, opt) # val_loader is None if phase is test.
    for phase, dataset_opt in opt['datasets'].items():
        data_sampler = None
        worker_init_fn = partial(Util.set_seed, gl_seed=opt['seed'])
        dataloader_args = dataset_opt['dataloader']['args']
        if phase == 'validation':
            val_set = init_obj(dataset_opt['which_dataset'], phase_logger, default_file_name='data.dataset', init_type='Dataset')
            val_loader = DataLoader(val_set, sampler=data_sampler, worker_init_fn=worker_init_fn, **dataloader_args)
        elif phase == 'test':
            test_set = init_obj(dataset_opt['which_dataset'], phase_logger, default_file_name='data.dataset', init_type='Dataset')
            test_loader = DataLoader(test_set, sampler=data_sampler, worker_init_fn=worker_init_fn, **dataloader_args)
    # endregion

    sampled_val_lambda_hat = run_validation(device, val_loader)
    run_test(wandb_logger, device, test_loader, sampled_val_lambda_hat)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/test_finetune_bounds_inpainting_center_dm_sba.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['calibration', 'validation', 'test'], help='Run train or test', default='calibration')
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)
    parser.add_argument('--n_soch_samples', type=int, default=50)
    parser.add_argument('--quantile', type=float, default=0.95)
    parser.add_argument('--enable_wandb', action='store_true')
    parser.add_argument('--clip_denoised', action='store_true')
    parser.add_argument('--finetune_loss', type=str, choices=['l2', 'quantile_regression'], default='l2')
    parser.add_argument('--distributed_worker_id', type=int, default=None)
    parser.add_argument('--prediction_time_step', type=int, default=155)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-5)

    args = parser.parse_args()
    opt = Praser.parse(args)

    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str

    opt['world_size'] = 1
    main_worker(0, opt)