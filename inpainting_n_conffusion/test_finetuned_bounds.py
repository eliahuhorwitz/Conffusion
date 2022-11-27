import argparse
import os
from functools import partial

import core.praser as Praser
import core.util as Util
import torch
from core import vis_util
from core.eval_utils_masked import get_rcps_metrics
from core.logger import InfoLogger, VisualWriter
from core.praser import init_obj
from core.wandb_logger import WandbLogger
from data import define_dataloader
from models import create_model, define_loss, define_metric, define_network
from models.conffusion.add_bounds import Conffusion
from torch.utils.data import DataLoader


def run_test(diffusion_with_bounds, wandb_logger, device, test_step, test_loader):
    with torch.no_grad():
        test_batch_size = test_loader.batch_size
        dataset_len = len(test_loader.dataset)


        res = 256
        n_img_channels = 3
        all_test_pred_lower_bounds = torch.zeros((min(dataset_len, len(test_loader) * test_batch_size), n_img_channels, res, res), device=device)
        all_test_pred_upper_bounds = torch.zeros((min(dataset_len, len(test_loader) * test_batch_size), n_img_channels, res, res), device=device)
        all_test_gt_samples = torch.zeros((min(dataset_len, len(test_loader) * test_batch_size), n_img_channels, res, res), device=device)
        all_test_maskes = torch.zeros((min(dataset_len, len(test_loader) * test_batch_size), n_img_channels, res, res), device=device)
        all_test_partial_gt_images = torch.zeros((min(dataset_len, len(test_loader) * test_batch_size), n_img_channels, res, res))
        all_test_masked_samples = torch.zeros((min(dataset_len, len(test_loader) * test_batch_size), n_img_channels, res, res))

        for test_idx, test_data in enumerate(test_loader):

            if opt['train']['finetune_loss'] == 'quantile_regression':
                test_masked_image = test_data["cond_image"].to(device)
                test_sampled_mask = test_data["mask"].to(device)
            else:
                test_masked_image = test_data["masked_samples"].to(device)
                test_sampled_mask = test_data["sampled_masks"].to(device)


            test_gt_image = test_data["gt_image"].to(device)


            test_pred_lower_bound, test_pred_upper_bound = diffusion_with_bounds(test_masked_image)

            test_pred_lower_bound = (torch.zeros_like(test_masked_image) * (1. - test_sampled_mask) + test_sampled_mask * test_pred_lower_bound)
            test_pred_upper_bound = (torch.zeros_like(test_masked_image) * (1. - test_sampled_mask) + test_sampled_mask * test_pred_upper_bound)
            partial_gt = (torch.zeros_like(test_masked_image) * (1. - test_sampled_mask) + test_sampled_mask * test_gt_image)

            # region HOLD ON TO THE SAMPLE
            start_idx = test_idx * test_batch_size
            end_idx = min(dataset_len, ((test_idx + 1) * test_batch_size))
            all_test_pred_lower_bounds[start_idx:end_idx] = test_pred_lower_bound
            all_test_pred_upper_bounds[start_idx:end_idx] = test_pred_upper_bound
            all_test_gt_samples[start_idx:end_idx] = test_gt_image
            all_test_partial_gt_images[start_idx:end_idx] = partial_gt
            all_test_maskes[start_idx:end_idx] = test_sampled_mask.detach().cpu()
            all_test_masked_samples[start_idx:end_idx] = test_masked_image.detach().cpu()
            # endregion

        # region NORM TO [0,1]
        # Transform all tensors to [0,1] for calibration and metric calculation
        all_test_gt_samples = Util.normalize_tensor(all_test_gt_samples)
        all_test_pred_lower_bounds = Util.normalize_tensor(all_test_pred_lower_bounds)
        all_test_pred_upper_bounds = Util.normalize_tensor(all_test_pred_upper_bounds)
        all_test_partial_gt_images = Util.normalize_tensor(all_test_partial_gt_images)
        all_test_masked_samples = Util.normalize_tensor(all_test_masked_samples)
        # endregion

        # region MULTIPLY BOUNDS BY LAMBDA
        pred_test_calibrated_l = (all_test_pred_lower_bounds / diffusion_with_bounds.pred_lambda_hat).clamp(0., 1.)
        pred_test_calibrated_u = (all_test_pred_upper_bounds * diffusion_with_bounds.pred_lambda_hat).clamp(0., 1.)
        # endregion

        calibrated_pred_risks_losses, calibrated_pred_sizes_mean, calibrated_pred_sizes_median, calibrated_pred_stratified_risks = get_rcps_metrics(pred_test_calibrated_l, pred_test_calibrated_u, all_test_gt_samples, all_test_maskes)

        if wandb_logger:
            image_grid = vis_util.create_image_grid(pred_test_calibrated_l.detach().cpu() * 2 - 1,
                                           pred_test_calibrated_u.detach().cpu() * 2 - 1,
                                           all_test_partial_gt_images.detach().cpu() * 2 - 1,
                                           all_test_masked_samples.detach().cpu() * 2 - 1,
                                           all_test_gt_samples.detach().cpu() * 2 - 1)

            wandb_logger.log_image("Test/Images", image_grid, caption="Pred L, Pred U, In LR, GT HR", commit=False)

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




def main_worker(gpu, opt):

    # region INIT SEEDS
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu
    torch.backends.cudnn.enabled = True
    Util.set_seed(opt['seed'])

    device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
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
        if phase == 'test':
            test_set = init_obj(dataset_opt['which_dataset'], phase_logger, default_file_name='data.dataset', init_type='Dataset')
            test_loader = DataLoader(test_set, sampler=data_sampler, worker_init_fn=worker_init_fn, **dataloader_args)
    # endregion

    # region INIT MODEL
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]
    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

    model = create_model(opt=opt, networks=networks, phase_loader=test_loader, val_loader=val_loader, losses=losses, metrics=metrics, logger=phase_logger, writer=phase_writer, wandb_logger=wandb_logger)
    model.netG.set_new_noise_schedule(phase=opt['phase'])

    diffusion_with_bounds = Conffusion(model, opt, device, load_finetuned=args.phase == "test").to(device)
    optimizer = torch.optim.Adam(list(diffusion_with_bounds.parameters()), lr=opt['train']['lr'])


    diffusion_with_bounds.eval()
    run_test(diffusion_with_bounds, wandb_logger, device, test_step, test_loader)
    # endregion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/test_finetune_bounds_inpainting_center_nconffusion.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['calibration', 'validation', 'test'], help='Run train or test', default='test')
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