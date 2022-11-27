import argparse
import logging

import core.logger as Logger
import core.metrics as Metrics
import data as Data
import torch
from core import vis_util
from core.calibration import calibrate_model
from core.eval_utils import get_rcps_metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter


def run_validation(opt, device, val_loader):
    with torch.no_grad():
        # region INIT VARS
        val_batch_size = val_loader.batch_size
        dataset_len = len(val_loader.dataset)

        sr_res = opt['datasets']['validation']['r_resolution']
        n_img_channels = 3
        all_val_sampled_lower_bounds = torch.zeros((min(dataset_len, len(val_loader) * val_batch_size), n_img_channels, sr_res, sr_res), device=device)
        all_val_sampled_upper_bounds = torch.zeros((min(dataset_len, len(val_loader) * val_batch_size), n_img_channels, sr_res, sr_res), device=device)
        all_val_gt_samples = torch.zeros((min(dataset_len, len(val_loader) * val_batch_size), n_img_channels, sr_res, sr_res), device=device)
        # endregion

        for val_idx, val_data in enumerate(val_loader):
            val_sampled_l_bound = val_data["lower_bound"].to(device)
            val_sampled_u_bound = val_data["upper_bound"].to(device)
            val_gt_image = val_data['HR'].to(device)

            # region HOLD ON TO THE SAMPLE
            start_idx = val_idx * val_batch_size
            end_idx = min(dataset_len, ((val_idx + 1) * val_batch_size))
            all_val_sampled_lower_bounds[start_idx:end_idx] = val_sampled_l_bound
            all_val_sampled_upper_bounds[start_idx:end_idx] = val_sampled_u_bound
            all_val_gt_samples[start_idx:end_idx] = val_gt_image
            #endregion

        # region NORM TO [0,1]
        # Transform all tensors to [0,1] for calibration and metric calculation
        all_val_sampled_lower_bounds = Metrics.normalize_tensor(all_val_sampled_lower_bounds)
        all_val_sampled_upper_bounds = Metrics.normalize_tensor(all_val_sampled_upper_bounds)
        all_val_gt_samples = Metrics.normalize_tensor(all_val_gt_samples)
        # endregion

        sampled_val_lambda_hat, _, _ = calibrate_model(all_val_sampled_lower_bounds, all_val_sampled_upper_bounds, all_val_gt_samples)

        return sampled_val_lambda_hat



def run_test(wandb_logger, device, test_loader, sampled_val_lambda_hat):
    with torch.no_grad():
        # region INIT VARS
        test_batch_size = test_loader.batch_size
        dataset_len = len(test_loader.dataset)


        sr_res = opt['datasets']['test']['r_resolution']
        n_img_channels = 3
        all_test_sampled_lower_bounds = torch.zeros((min(dataset_len, len(test_loader) * test_batch_size), n_img_channels, sr_res, sr_res), device=device)
        all_test_sampled_upper_bounds = torch.zeros((min(dataset_len, len(test_loader) * test_batch_size), n_img_channels, sr_res, sr_res), device=device)
        all_test_gt_samples = torch.zeros((min(dataset_len, len(test_loader) * test_batch_size), n_img_channels, sr_res, sr_res), device=device)
        all_test_lr_samples = torch.zeros((min(dataset_len, len(test_loader) * test_batch_size), n_img_channels, sr_res, sr_res), device=device)
        # endregion

        for test_idx, test_data in enumerate(test_loader):
            test_low_res_image = test_data["SR"].to(device)
            test_sampled_l_bound = test_data["lower_bound"].to(device)
            test_sampled_u_bound = test_data["upper_bound"].to(device)
            test_gt_image = test_data['HR'].to(device)



            # region HOLD ON TO THE SAMPLE
            start_idx = test_idx * test_batch_size
            end_idx = min(dataset_len, ((test_idx + 1) * test_batch_size))
            all_test_sampled_lower_bounds[start_idx:end_idx] = test_sampled_l_bound
            all_test_sampled_upper_bounds[start_idx:end_idx] = test_sampled_u_bound
            all_test_gt_samples[start_idx:end_idx] = test_gt_image
            all_test_lr_samples[start_idx:end_idx] = test_low_res_image

            # endregion

        # region NORM TO [0,1]
        # Transform all tensors to [0,1] for calibration and metric calculation
        all_test_sampled_lower_bounds = Metrics.normalize_tensor(all_test_sampled_lower_bounds)
        all_test_sampled_upper_bounds = Metrics.normalize_tensor(all_test_sampled_upper_bounds)
        all_test_gt_samples = Metrics.normalize_tensor(all_test_gt_samples)
        all_test_lr_samples = Metrics.normalize_tensor(all_test_lr_samples) 
        # endregion

        # region MULTIPLY BOUNDS BY LAMBDA
        sampled_test_calibrated_l = (all_test_sampled_lower_bounds / sampled_val_lambda_hat).clamp(0., 1.)
        sampled_test_calibrated_u = (all_test_sampled_upper_bounds * sampled_val_lambda_hat).clamp(0., 1.)
        # endregion


        calibrated_sampled_risks_losses, calibrated_sampled_sizes_mean, calibrated_sampled_sizes_median, calibrated_sampled_stratified_risks = get_rcps_metrics(sampled_test_calibrated_l, sampled_test_calibrated_u, all_test_gt_samples)


        if wandb_logger:
            image_grid = vis_util.create_image_grid(sampled_test_calibrated_l.detach().cpu() * 2 - 1,
                                           sampled_test_calibrated_u.detach().cpu() * 2 - 1,
                                           all_test_lr_samples.detach().cpu() * 2 - 1,
                                           all_test_gt_samples.detach().cpu() * 2 - 1)

            wandb_logger.log_image("Test/Images", image_grid, caption="Sampled L, Sampled U, LR, GT SR", commit=False)

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/test_finetune_bounds_16_128_conffusion.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['calibration', 'validation', 'test'], default='test')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('--enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('--finetune_loss', type=str, choices=['l2', 'quantile_regression'], default='l2')
    parser.add_argument('--distributed_worker_id', type=int, default=None)
    parser.add_argument('--prediction_time_step', type=int, default=0)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # region INIT LOGGER
    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])
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


    # region INIT DATASETS


    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'validation':
            val_set = Data.create_dataset_for_bounds_finetuning(dataset_opt, phase)
            val_loader = Data.create_dataloader(val_set, dataset_opt, phase)
        elif phase == 'test':
            test_set = Data.create_dataset_for_bounds_finetuning(dataset_opt, phase)
            test_loader = Data.create_dataloader(test_set, dataset_opt, phase)


    # endregion

    sampled_val_lambda_hat = run_validation(opt, device, val_loader)
    run_test(wandb_logger, device, test_loader, sampled_val_lambda_hat)