import argparse
import os

import conffusion.metrics as Metrics
import sr_dataset as Data
import torch
import wandb
from conffusion import vis_util
from conffusion.add_bounds import Conffusion
from conffusion.eval_utils import get_rcps_metrics
from conffusion.wandb_logger import WandbLogger
from guided_diffusion import dist_util, logger, script_util
from guided_diffusion.script_util import (add_dict_to_argparser, args_to_dict,
                                          create_model_and_diffusion,
                                          model_and_diffusion_defaults)
from guided_diffusion.fp16_util import MixedPrecisionTrainer

def run_test(diffusion_with_bounds, wandb_logger, device, val_step, test_loader):
    with torch.no_grad():
        # region INIT VARS
        test_batch_size = test_loader.batch_size
        dataset_len = len(test_loader.dataset)

        sr_res = 128
        n_img_channels = 3
        all_test_pred_lower_bounds = torch.zeros((min(dataset_len, len(test_loader) * test_batch_size), n_img_channels, sr_res, sr_res), device=device)
        all_test_pred_upper_bounds = torch.zeros((min(dataset_len, len(test_loader) * test_batch_size), n_img_channels, sr_res, sr_res), device=device)
        all_test_gt_samples = torch.zeros((min(dataset_len, len(test_loader) * test_batch_size), n_img_channels, sr_res, sr_res), device=device)
        all_test_lr_samples = torch.zeros((min(dataset_len, len(test_loader) * test_batch_size), n_img_channels, sr_res, sr_res), device=device)
        # endregion

        for test_idx, test_data in enumerate(test_loader):
            test_low_res_image = test_data["SR"].to(device)
            test_gt_image = test_data['HR'].to(device)

            test_pred_lower_bound, test_pred_upper_bound, _ = diffusion_with_bounds(test_low_res_image)

            # region HOLD ON TO THE SAMPLE
            start_idx = test_idx * test_batch_size
            end_idx = min(dataset_len, ((test_idx + 1) * test_batch_size))
            all_test_pred_lower_bounds[start_idx:end_idx] = test_pred_lower_bound
            all_test_pred_upper_bounds[start_idx:end_idx] = test_pred_upper_bound
            all_test_gt_samples[start_idx:end_idx] = test_gt_image
            all_test_lr_samples[start_idx:end_idx] = test_low_res_image
            #endregion

        # region NORM TO [0,1]
        # Transform all tensors to [0,1] for calibration and metric calculation
        all_test_gt_samples = Metrics.normalize_tensor(all_test_gt_samples)
        all_test_pred_lower_bounds = Metrics.normalize_tensor(all_test_pred_lower_bounds)
        all_test_pred_upper_bounds = Metrics.normalize_tensor(all_test_pred_upper_bounds)
        all_test_lr_samples = Metrics.normalize_tensor(all_test_lr_samples)
        # endregion

        pred_test_calibrated_l = (all_test_pred_lower_bounds / diffusion_with_bounds.pred_lambda_hat).clamp(0., 1.)
        pred_test_calibrated_u = (all_test_pred_upper_bounds * diffusion_with_bounds.pred_lambda_hat).clamp(0., 1.)
        calibrated_pred_risks_losses, calibrated_pred_sizes_mean, calibrated_pred_sizes_median, calibrated_pred_stratified_risks = get_rcps_metrics(pred_test_calibrated_l, pred_test_calibrated_u, all_test_gt_samples)


        # region LOG TO WANDB
        if wandb_logger:
            image_grid = vis_util.create_image_grid_sr(pred_test_calibrated_l.detach().cpu() * 2 - 1,
                                           pred_test_calibrated_u.detach().cpu() * 2 - 1,
                                           all_test_lr_samples.detach().cpu() * 2 - 1,
                                           all_test_gt_samples.detach().cpu() * 2 - 1)
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
            wandb_logger.log_metrics(calibrated_pred_stratified_risks_data, commit=True)
        # endregion


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()

    experiments_root = script_util.get_loggging_path(args)

    logger.configure(dir=experiments_root)

    wandb_logger = WandbLogger(args)
    wandb.define_metric('validation/val_step')
    wandb.define_metric('Finetune/finetune_step')
    wandb.define_metric('epoch')
    wandb.define_metric("validation/*", step_metric="val_step")
    wandb.define_metric("Finetune/*", step_metric="finetune_step")

    wandb.define_metric("Validation/Calibrated Pred Size Mean", summary="min")
    wandb.define_metric("Calibrated Pred Size Mean", summary="min")
    wandb.define_metric("Validation/Calibrated Pred Size Median", summary="min")
    wandb.define_metric("Calibrated Pred Size Median", summary="min")



    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()), finetune_bounds=True)

    orig_state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    rectified_state_dict = {}
    for k, v in orig_state_dict.items():
        if k.startswith('model.'):
            rectified_state_dict[k.replace('model.', '')] = v
        else:
            rectified_state_dict[k] = v
    model.load_state_dict(rectified_state_dict)



    model.to(dist_util.dev())
    diffusion_with_bounds = Conffusion(diffusion, model, dist_util.dev(), load_finetuned=True, prediction_time_step=args.prediction_time_step, options_path=args.options_path).to(dist_util.dev())

    diffusion_with_bounds.eval()
    dist_util.sync_params(model.parameters())
    model.eval()
    MixedPrecisionTrainer(model=diffusion_with_bounds, use_fp16=True, fp16_scale_growth=1e-3, )



    test_set = Data.create_dataset(args.dataset_root, "test")
    test_loader = Data.create_dataloader(test_set, "test")

    run_test(diffusion_with_bounds, wandb_logger, dist_util.dev(), 0, test_loader)




def create_argparser():
    defaults = dict(clip_denoised=True, num_samples=10000, batch_size=16, use_ddim=False, model_path="",)
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--prediction_time_step', type=int, default=15)
    parser.add_argument('--options_path', default=None)
    parser.add_argument('--dataset_root', type=str, default="../datasets/celebahq_16_128")
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
