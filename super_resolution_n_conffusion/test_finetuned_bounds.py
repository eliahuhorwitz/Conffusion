import argparse
import logging

import core.logger as Logger
import core.metrics as Metrics
import data as Data
import model as Model
import torch
from core import vis_util
from core.eval_utils import get_rcps_metrics
from core.wandb_logger import WandbLogger
from model.conffusion.add_bounds import Conffusion
from tensorboardX import SummaryWriter


def run_test(diffusion_with_bounds, wandb_logger, device, test_step, test_loader):
    with torch.no_grad():
        # region INIT VARS
        test_batch_size = test_loader.batch_size
        dataset_len = len(test_loader.dataset)


        sr_res = opt['datasets']['test']['r_resolution']
        n_img_channels = 3
        all_test_pred_lower_bounds = torch.zeros((min(dataset_len, len(test_loader) * test_batch_size), n_img_channels, sr_res, sr_res), device=device)
        all_test_pred_upper_bounds = torch.zeros((min(dataset_len, len(test_loader) * test_batch_size), n_img_channels, sr_res, sr_res), device=device)
        all_test_gt_samples = torch.zeros((min(dataset_len, len(test_loader) * test_batch_size), n_img_channels, sr_res, sr_res), device=device)
        all_test_lr_samples = torch.zeros((min(dataset_len, len(test_loader) * test_batch_size), n_img_channels, sr_res, sr_res), device=device)
        # endregion

        for test_idx, test_data in enumerate(test_loader):
            test_low_res_image = test_data["SR"].to(device)
            test_gt_image = test_data['HR'].to(device)

            test_pred_lower_bound, test_pred_upper_bound = diffusion_with_bounds(test_low_res_image)


            # region HOLD ON TO THE SAMPLE
            start_idx = test_idx * test_batch_size
            end_idx = min(dataset_len, ((test_idx + 1) * test_batch_size))
            all_test_pred_lower_bounds[start_idx:end_idx] = test_pred_lower_bound
            all_test_pred_upper_bounds[start_idx:end_idx] = test_pred_upper_bound
            all_test_gt_samples[start_idx:end_idx] = test_gt_image
            all_test_lr_samples[start_idx:end_idx] = test_low_res_image

            # endregion

        # region NORM TO [0,1]
        # Transform all tensors to [0,1] for calibration and metric calculation
        all_test_gt_samples = Metrics.normalize_tensor(all_test_gt_samples)
        all_test_pred_lower_bounds = Metrics.normalize_tensor(all_test_pred_lower_bounds)
        all_test_pred_upper_bounds = Metrics.normalize_tensor(all_test_pred_upper_bounds)
        all_test_lr_samples = Metrics.normalize_tensor(all_test_lr_samples) 
        # endregion

        # region MULTIPLY BOUNDS BY LAMBDA
        pred_test_calibrated_l = (all_test_pred_lower_bounds / diffusion_with_bounds.pred_lambda_hat).clamp(0., 1.)
        pred_test_calibrated_u = (all_test_pred_upper_bounds * diffusion_with_bounds.pred_lambda_hat).clamp(0., 1.)
        # endregion

        calibrated_pred_risks_losses, calibrated_pred_sizes_mean, calibrated_pred_sizes_median, calibrated_pred_stratified_risks = get_rcps_metrics(pred_test_calibrated_l, pred_test_calibrated_u, all_test_gt_samples)

        if wandb_logger:
            image_grid = vis_util.create_image_grid(pred_test_calibrated_l.detach().cpu() * 2 - 1,
                                           pred_test_calibrated_u.detach().cpu() * 2 - 1,
                                           all_test_lr_samples.detach().cpu() * 2 - 1,
                                           all_test_gt_samples.detach().cpu() * 2 - 1)

            wandb_logger.log_image("Test/Images", image_grid, caption="Pred L, Pred U, LR, GT SR", commit=False)

            wandb_logger.log_metrics({'Test/Calibrated Pred Risk': calibrated_pred_risks_losses, 'Test/test_step': test_step}, commit=False)
            wandb_logger.log_metrics({'Test/Calibrated Pred Size Mean': calibrated_pred_sizes_mean, 'Test/test_step': test_step}, commit=False)
            wandb_logger.log_metrics({'Test/Calibrated Pred Size Median': calibrated_pred_sizes_median, 'Test/test_step': test_step},commit=False)
            calibrated_pred_stratified_risks_data = dict([[label, test] for (label, test) in
                                                          zip([
                                                                  "Test/Calibrated Pred Stratified Risks - Short",
                                                                  "Test/Calibrated Pred Stratified Risks - Short-Medium",
                                                                  "Test/Calibrated Pred Stratified Risks - Medium-Long",
                                                                  "Test/Calibrated Pred Stratified Risks - Long"],
                                                              calibrated_pred_stratified_risks.detach().cpu().numpy())])
            calibrated_pred_stratified_risks_data['Test/test_step'] = test_step
            wandb_logger.log_metrics(calibrated_pred_stratified_risks_data)



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
    test_step = 0
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('test/test_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        wandb.define_metric("test/*", step_metric="test_step")
        wandb.define_metric("test/Calibrated Pred Size Mean", summary="min")
        wandb.define_metric("test/Calibrated Pred Size Median", summary="min")
        wandb.define_metric("Calibrated Pred Size Mean", summary="min")
        wandb.define_metric("Calibrated Pred Size Median", summary="min")
    else:
        wandb_logger = None
    # endregion


    # region INIT DATASETS
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test':
            test_set = Data.create_dataset(dataset_opt, phase)
            test_loader = Data.create_dataloader(test_set, dataset_opt, phase)
    # endregion

    # region INIT MODEL
    diffusion = Model.create_model(opt)

    diffusion.set_new_noise_schedule(opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    diffusion_with_bounds = Conffusion(diffusion, opt, device, load_finetuned=args.phase == "test").to(device)
    optimizer = torch.optim.Adam(list(diffusion_with_bounds.parameters()), lr=opt['train']["optimizer"]["lr"])
    # endregion

    diffusion_with_bounds.eval()
    run_test(diffusion_with_bounds, wandb_logger, device, test_step, test_loader)