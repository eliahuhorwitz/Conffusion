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
from models import create_model, define_loss, define_metric, define_network
from models.conffusion.add_bounds import Conffusion
from torch.utils.data import DataLoader


def run_validation(opt, diffusion_with_bounds, wandb_logger, device, val_step, val_loader):
    print("Starting Validation...")
    with torch.no_grad():
        # region INIT VARS
        val_loss = 0
        val_batch_size = val_loader.batch_size
        dataset_len = len(val_loader.dataset)

        sr_res = 256
        n_img_channels = 3
        all_val_pred_lower_bounds = torch.zeros((min(dataset_len, len(val_loader) * val_batch_size), n_img_channels, sr_res, sr_res), device=device)
        all_val_pred_upper_bounds = torch.zeros((min(dataset_len, len(val_loader) * val_batch_size), n_img_channels, sr_res, sr_res), device=device)
        all_val_gt_samples = torch.zeros((min(dataset_len, len(val_loader) * val_batch_size), n_img_channels, sr_res, sr_res), device=device)
        all_val_maskes = torch.zeros((min(dataset_len, len(val_loader) * val_batch_size), n_img_channels, sr_res, sr_res), device=device)
        all_val_partial_gt_images = torch.zeros((min(dataset_len, len(val_loader) * val_batch_size), n_img_channels, sr_res, sr_res))
        all_val_masked_samples = torch.zeros((min(dataset_len, len(val_loader) * val_batch_size), n_img_channels, sr_res, sr_res))
        # endregion

        for val_idx, val_data in enumerate(val_loader):

            val_gt_image = val_data["gt_image"].to(device)

            if opt['train']['finetune_loss'] == 'quantile_regression':
                val_masked_image = val_data["cond_image"].to(device)
                val_mask = val_data["mask"].to(device)
            else:
                val_masked_image = val_data["masked_samples"].to(device)
                val_mask = val_data["sampled_masks"].to(device)

            val_pred_lower_bound, val_pred_upper_bound = diffusion_with_bounds(val_masked_image)

            val_pred_lower_bound = (torch.zeros_like(val_masked_image) * (1. - val_mask) + val_mask * val_pred_lower_bound)
            val_pred_upper_bound = (torch.zeros_like(val_masked_image) * (1. - val_mask) + val_mask * val_pred_upper_bound)
            partial_gt = (torch.zeros_like(val_masked_image) * (1. - val_mask) + val_mask * val_gt_image)

            if opt['train']['finetune_loss'] == 'quantile_regression':
                val_bounds_loss = diffusion_with_bounds.quantile_regression_loss_fn(val_pred_lower_bound, val_pred_upper_bound, val_gt_image)
            else:
                val_sampled_l_bound = val_data["lower_bound"].to(device)
                val_sampled_u_bound = val_data["upper_bound"].to(device)
                val_sampled_l_bound = (torch.zeros_like(val_sampled_l_bound) * (1. - val_mask) + val_mask * val_sampled_l_bound)
                val_sampled_u_bound = (torch.zeros_like(val_sampled_u_bound) * (1. - val_mask) + val_mask * val_sampled_u_bound)
                val_bounds_loss = diffusion_with_bounds.bounds_regression_loss_fn(val_pred_lower_bound, val_pred_upper_bound, val_sampled_l_bound, val_sampled_u_bound)
            val_loss += val_bounds_loss.item()

            # region HOLD ON TO THE SAMPLE
            start_idx = val_idx * val_batch_size
            end_idx = min(dataset_len, ((val_idx + 1) * val_batch_size))
            all_val_pred_lower_bounds[start_idx:end_idx] = val_pred_lower_bound
            all_val_pred_upper_bounds[start_idx:end_idx] = val_pred_upper_bound
            all_val_gt_samples[start_idx:end_idx] = val_gt_image
            all_val_maskes[start_idx:end_idx] = val_mask.detach().cpu()
            all_val_partial_gt_images[start_idx:end_idx] = partial_gt
            all_val_masked_samples[start_idx:end_idx] = val_masked_image.detach().cpu()
            #endregion

        # region NORM TO [0,1]
        # Transform all tensors to [0,1] for calibration and metric calculation
        all_val_gt_samples = Util.normalize_tensor(all_val_gt_samples)
        all_val_pred_lower_bounds = Util.normalize_tensor(all_val_pred_lower_bounds)
        all_val_pred_upper_bounds = Util.normalize_tensor(all_val_pred_upper_bounds)
        all_val_partial_gt_images = Util.normalize_tensor(all_val_partial_gt_images)
        all_val_masked_samples = Util.normalize_tensor(all_val_masked_samples)
        # endregion


        # region CALIBRATION
        pred_val_lambda_hat, pred_val_calibrated_l, pred_val_calibrated_u = calibrate_model(all_val_pred_lower_bounds, all_val_pred_upper_bounds, all_val_gt_samples, all_val_maskes)
        calibrated_pred_risks_losses, calibrated_pred_sizes_mean, calibrated_pred_sizes_median, calibrated_pred_stratified_risks = get_rcps_metrics(pred_val_calibrated_l, pred_val_calibrated_u, all_val_gt_samples, all_val_maskes)

        # endregion

        # region LOG TO WANDB
        if wandb_logger:
            image_grid = vis_util.create_image_grid(pred_val_calibrated_l.detach().cpu() * 2 - 1,
                                           pred_val_calibrated_u.detach().cpu() * 2 - 1,
                                           all_val_partial_gt_images.detach().cpu() * 2 - 1,
                                           all_val_masked_samples.detach().cpu() * 2 - 1,
                                           all_val_gt_samples.detach().cpu() * 2 - 1)
            wandb_logger.log_image("Validation/Images", image_grid, caption="Pred L, Pred U, GT, Masked Input, Full GT", commit=False)

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
    best_interval_size = float('inf')
    val_step = 0
    current_step = 0
    current_epoch = 0
    n_iter = opt['train']['n_iter']
    while current_step < n_iter:
        current_epoch += 1
        for train_idx, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > n_iter:
                break

            if opt['train']['finetune_loss'] == 'quantile_regression':
                masked_image = train_data["cond_image"].to(device)
                mask = train_data["mask"].to(device)
            else:
                masked_image = train_data["masked_samples"].to(device)
                mask = train_data["sampled_masks"].to(device)

            gt_image = train_data["gt_image"].to(device)

            pred_lower_bound, pred_upper_bound = diffusion_with_bounds(masked_image)
            pred_lower_bound = (torch.zeros_like(masked_image) * (1. - mask) + mask * pred_lower_bound)
            pred_upper_bound = (torch.zeros_like(masked_image) * (1. - mask) + mask * pred_upper_bound)
            partial_gt = (torch.zeros_like(gt_image) * (1. - mask) + mask * gt_image)
            optimizer.zero_grad()


            if opt['train']['finetune_loss'] == 'quantile_regression':
                bounds_loss = diffusion_with_bounds.quantile_regression_loss_fn(pred_lower_bound, pred_upper_bound, partial_gt)
            else:
                sampled_l_bound = train_data["lower_bound"].to(device)
                sampled_u_bound = train_data["upper_bound"].to(device)
                sampled_l_bound = (torch.zeros_like(masked_image) * (1. - mask) + mask * sampled_l_bound)
                sampled_u_bound = (torch.zeros_like(masked_image) * (1. - mask) + mask * sampled_u_bound)
                bounds_loss = diffusion_with_bounds.bounds_regression_loss_fn(pred_lower_bound, pred_upper_bound, sampled_l_bound, sampled_u_bound)


            diffusion_with_bounds.log_dict['Finetune/Loss'] = bounds_loss.item()
            bounds_loss.backward()
            optimizer.step()

            if current_step % opt['train']['print_freq'] == 0 or current_step - 1 == 0:
                vis_util.log_train(diffusion_with_bounds, wandb_logger, pred_lower_bound, pred_upper_bound, partial_gt, train_data)

            if current_step % opt['train']['val_freq'] == 0 or current_step - 1 == 0:
                calibrated_pred_risks_losses, calibrated_pred_sizes, pred_val_lambda_hat = run_validation(opt, diffusion_with_bounds, wandb_logger, device, val_step, val_loader)
                val_step += 1
                if calibrated_pred_risks_losses < 0.1 and calibrated_pred_sizes < best_interval_size:
                    print(f'Saving best model and training states, interval size is {calibrated_pred_sizes}, lambda_hat is {pred_val_lambda_hat}')
                    diffusion_with_bounds.save_best_network(current_epoch, current_step, optimizer, pred_lambda_hat=pred_val_lambda_hat)
                    best_interval_size = calibrated_pred_sizes

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

    # region INIT LOGGER
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)  
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))
    # endregion

    # region INIT DATASETS
    # phase_loader, val_loader = define_dataloader(phase_logger, opt) # val_loader is None if phase is test.
    for phase, dataset_opt in opt['datasets'].items():
        data_sampler = None
        worker_init_fn = partial(Util.set_seed, gl_seed=opt['seed'])
        dataloader_args = dataset_opt['dataloader']['args']
        if phase == 'calibration' and args.phase != 'validation':
            train_set = init_obj(dataset_opt['which_dataset'], phase_logger, default_file_name='data.dataset', init_type='Dataset')
            train_loader = DataLoader(train_set, sampler=data_sampler, worker_init_fn=worker_init_fn, **dataloader_args)
        elif phase == 'validation':
            val_set = init_obj(dataset_opt['which_dataset'], phase_logger, default_file_name='data.dataset', init_type='Dataset')
            val_loader = DataLoader(val_set, sampler=data_sampler, worker_init_fn=worker_init_fn, **dataloader_args)
    # endregion

    # region INIT MODEL
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]
    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

    model = create_model(opt=opt, networks=networks, phase_loader=train_loader, val_loader=val_loader, losses=losses, metrics=metrics, logger=phase_logger, writer=phase_writer, wandb_logger=wandb_logger)
    model.netG.set_new_noise_schedule(phase=opt['phase'])

    diffusion_with_bounds = Conffusion(model, opt, device, load_finetuned=args.phase == "test").to(device)
    optimizer = torch.optim.Adam(list(diffusion_with_bounds.parameters()), lr=opt['train']['lr'])


    diffusion_with_bounds.train()
    # endregion

    run_training(opt, diffusion_with_bounds, wandb_logger, device, optimizer, train_loader, val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/finetune_bounds_inpainting_center_celebahq.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['calibration', 'validation', 'test'], help='Run train or test', default='calibration')
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)
    parser.add_argument('--n_soch_samples', type=int, default=50)
    parser.add_argument('--quantile', type=float, default=0.95)
    parser.add_argument('--enable_wandb', action='store_true')
    parser.add_argument('--clip_denoised', action='store_true')
    parser.add_argument('--finetune_loss', type=str, choices=['l2', 'quantile_regression'], default='quantile_regression')
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