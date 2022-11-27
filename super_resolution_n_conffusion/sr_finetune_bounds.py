import argparse
import logging

import core.logger as Logger
import core.metrics as Metrics
import data as Data
import model as Model
import torch
from core import vis_util
from core.calibration import calibrate_model
from core.eval_utils import get_rcps_metrics
from core.wandb_logger import WandbLogger
from model.conffusion.add_bounds import Conffusion
from tensorboardX import SummaryWriter


def run_validation(opt, diffusion_with_bounds, wandb_logger, device, val_step, val_loader):
    print("Starting Validation...")
    with torch.no_grad():
        # region INIT VARS
        val_loss = 0
        val_batch_size = val_loader.batch_size
        dataset_len = len(val_loader.dataset)

        sr_res = opt['datasets']['validation']['r_resolution']
        n_img_channels = 3
        all_val_pred_lower_bounds = torch.zeros((min(dataset_len, len(val_loader) * val_batch_size), n_img_channels, sr_res, sr_res), device=device)
        all_val_pred_upper_bounds = torch.zeros((min(dataset_len, len(val_loader) * val_batch_size), n_img_channels, sr_res, sr_res), device=device)
        all_val_gt_samples = torch.zeros((min(dataset_len, len(val_loader) * val_batch_size), n_img_channels, sr_res, sr_res), device=device)
        all_val_lr_samples = torch.zeros((min(dataset_len, len(val_loader) * val_batch_size), n_img_channels, sr_res, sr_res), device=device)
        # endregion

        for val_idx, val_data in enumerate(val_loader):
            val_low_res_image = val_data["SR"].to(device)
            val_gt_image = val_data['HR'].to(device)

            val_pred_lower_bound, val_pred_upper_bound = diffusion_with_bounds(val_low_res_image)

            if opt['train']['finetune_loss'] == 'quantile_regression':
                val_bounds_loss = diffusion_with_bounds.quantile_regression_loss_fn(val_pred_lower_bound, val_pred_upper_bound, val_gt_image)
            else:
                val_sampled_l_bound = val_data["lower_bound"].to(device)
                val_sampled_u_bound = val_data["upper_bound"].to(device)
                val_bounds_loss = diffusion_with_bounds.bounds_regression_loss_fn(val_pred_lower_bound, val_pred_upper_bound, val_sampled_l_bound, val_sampled_u_bound)
            val_loss += val_bounds_loss.item()

            # region HOLD ON TO THE SAMPLE
            start_idx = val_idx * val_batch_size
            end_idx = min(dataset_len, ((val_idx + 1) * val_batch_size))
            all_val_pred_lower_bounds[start_idx:end_idx] = val_pred_lower_bound
            all_val_pred_upper_bounds[start_idx:end_idx] = val_pred_upper_bound
            all_val_gt_samples[start_idx:end_idx] = val_gt_image
            all_val_lr_samples[start_idx:end_idx] = val_low_res_image
            #endregion

        # region NORM TO [0,1]
        # Transform all tensors to [0,1] for calibration and metric calculation
        all_val_gt_samples = Metrics.normalize_tensor(all_val_gt_samples)
        all_val_pred_lower_bounds = Metrics.normalize_tensor(all_val_pred_lower_bounds)
        all_val_pred_upper_bounds = Metrics.normalize_tensor(all_val_pred_upper_bounds)
        all_val_lr_samples = Metrics.normalize_tensor(all_val_lr_samples)
        # endregion


        # region CALIBRATION
        pred_val_lambda_hat, pred_val_calibrated_l, pred_val_calibrated_u = calibrate_model(all_val_pred_lower_bounds, all_val_pred_upper_bounds, all_val_gt_samples)
        calibrated_pred_risks_losses, calibrated_pred_sizes_mean, calibrated_pred_sizes_median, calibrated_pred_stratified_risks = get_rcps_metrics(pred_val_calibrated_l, pred_val_calibrated_u, all_val_gt_samples)
        # endregion

        # region LOG TO WANDB
        if wandb_logger:
            image_grid = vis_util.create_image_grid(pred_val_calibrated_l.detach().cpu() * 2 - 1,
                                           pred_val_calibrated_u.detach().cpu() * 2 - 1,
                                           all_val_lr_samples.detach().cpu() * 2 - 1,
                                           all_val_gt_samples.detach().cpu() * 2 - 1)
            wandb_logger.log_image("Validation/Images", image_grid, caption="Pred L, Pred U, LR, GT SR", commit=False)

            wandb_logger.log_metrics({'Validation/Calibrated Pred Risk': calibrated_pred_risks_losses, 'Validation/val_step': val_step}, commit=False)
            wandb_logger.log_metrics({'Validation/Calibrated Pred Size Mean': calibrated_pred_sizes_mean, 'Validation/val_step': val_step}, commit=False)
            wandb_logger.log_metrics({'Validation/Calibrated Pred Size Median': calibrated_pred_sizes_median, 'Validation/val_step': val_step}, commit=False)
            calibrated_pred_stratified_risks_data = dict([[label, val] for (label, val) in
                                                          zip([
                                                                  "Validation/Calibrated Pred Stratified Risks - Short",
                                                                  "Validation/Calibrated Pred Stratified Risks - Short-Medium",
                                                                  "Validation/Calibrated Pred Stratified Risks - Medium-Long",
                                                                  "Validation/Calibrated Pred Stratified Risks - Long"],
                                                              calibrated_pred_stratified_risks.detach().cpu().numpy())])
            calibrated_pred_stratified_risks_data['Validation/val_step'] = val_step
            wandb_logger.log_metrics(calibrated_pred_stratified_risks_data, commit=False)

            wandb_logger.log_metrics({'Validation/Pred Lambda': pred_val_lambda_hat, 'Validation/val_step': val_step}, commit=False)
            wandb_logger.log_metrics({'Validation/Loss': (val_loss / len(val_loader)), 'Validation/val_step': val_step})

            print(f"Finished Validation, calibrated pred size = {calibrated_pred_sizes_mean:.3f}...\n\n")
            return calibrated_pred_risks_losses, calibrated_pred_sizes_mean, pred_val_lambda_hat
        # endregion

def run_training(opt, diffusion_with_bounds, wandb_logger, device, optimizer, train_loader, val_loader):
    val_step = 0
    current_step = 0
    current_epoch = 0
    n_iter = opt['train']['n_iter']

    best_interval_size = float('inf')

    while current_step < n_iter:
        current_epoch += 1

        for train_idx, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > n_iter:
                break
            low_res_image = train_data["SR"].to(device)
            gt_image = train_data['HR'].to(device)

            pred_lower_bound, pred_upper_bound = diffusion_with_bounds(low_res_image)
            optimizer.zero_grad()

            if opt['train']['finetune_loss'] == 'quantile_regression':
                bounds_loss = diffusion_with_bounds.quantile_regression_loss_fn(pred_lower_bound, pred_upper_bound, gt_image)
            else:
                sampled_l_bound = train_data["lower_bound"].to(device)
                sampled_u_bound = train_data["upper_bound"].to(device)
                bounds_loss = diffusion_with_bounds.bounds_regression_loss_fn(pred_lower_bound, pred_upper_bound, sampled_l_bound, sampled_u_bound)


            diffusion_with_bounds.log_dict['Finetune/Loss'] = bounds_loss.item()
            bounds_loss.backward()
            optimizer.step()

            if current_step % opt['train']['print_freq'] == 0 or current_step - 1 == diffusion.begin_step:
                vis_util.log_train(diffusion_with_bounds, wandb_logger, pred_lower_bound, pred_upper_bound, train_data)

            # region VALIDATION
            if current_step % opt['train']['val_freq'] == 0 or current_step - 1 == diffusion.begin_step:
                calibrated_pred_risks_losses, calibrated_pred_sizes, pred_val_lambda_hat = run_validation(opt, diffusion_with_bounds, wandb_logger, device, val_step, val_loader)
                val_step += 1
                if calibrated_pred_risks_losses < 0.1 and calibrated_pred_sizes < best_interval_size:
                    logger.info(f'Saving best model and training states, interval size is {calibrated_pred_sizes}, lambda_hat is {pred_val_lambda_hat}')
                    diffusion_with_bounds.save_best_network(current_epoch, current_step, optimizer, pred_lambda_hat=pred_val_lambda_hat)
                    best_interval_size = calibrated_pred_sizes
            # endregion


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/finetune_bounds_16_128_conffusion.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['calibration', 'validation', 'test'], default='calibration')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('--enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('--finetune_loss', type=str, choices=['l2', 'quantile_regression'], default='quantile_regression')
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
        wandb.define_metric('validation/val_step')
        wandb.define_metric('Finetune/finetune_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        wandb.define_metric("Finetune/*", step_metric="finetune_step")

        wandb.define_metric("Validation/Calibrated Pred Size Mean", summary="min")
        wandb.define_metric("Calibrated Pred Size Mean", summary="min")
        wandb.define_metric("Validation/Calibrated Pred Size Median", summary="min")
        wandb.define_metric("Calibrated Pred Size Median", summary="min")



    else:
        wandb_logger = None
    # endregion


    # region INIT DATASETS
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'calibration' and args.phase != 'validation':
            if args.finetune_loss == "l2":
                train_set = Data.create_dataset_for_bounds_finetuning(dataset_opt, phase)
            elif args.finetune_loss == "quantile_regression":
                train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
        elif phase == 'validation':
            if args.finetune_loss == "l2":
                val_set = Data.create_dataset_for_bounds_finetuning(dataset_opt, phase)
            elif args.finetune_loss == "quantile_regression":
                val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(val_set, dataset_opt, phase)
    # endregion

    # region INIT MODEL
    diffusion = Model.create_model(opt)

    diffusion.set_new_noise_schedule(opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    diffusion_with_bounds = Conffusion(diffusion, opt, device, load_finetuned=args.phase == "test").to(device)
    optimizer = torch.optim.Adam(list(diffusion_with_bounds.parameters()), lr=opt['train']["optimizer"]["lr"])
    # endregion

    run_training(opt, diffusion_with_bounds, wandb_logger, device, optimizer, train_loader, val_loader)