import argparse
import os

import conffusion.metrics as Metrics
import torch
import wandb
from conffusion import vis_util
from conffusion.add_bounds import Conffusion
from conffusion.eval_utils_masked import get_rcps_metrics
from conffusion.wandb_logger import WandbLogger
from guided_diffusion import dist_util, logger, script_util
from guided_diffusion.script_util import (add_dict_to_argparser, args_to_dict,
                                          create_model_and_diffusion,
                                          model_and_diffusion_defaults)
from inpainting_dataset.dataset import InpaintDataset
from guided_diffusion.fp16_util import MixedPrecisionTrainer

def run_test(diffusion_with_bounds, wandb_logger, device, val_step, val_loader):
    with torch.no_grad():
        # region INIT VARS
        val_batch_size = val_loader.batch_size
        dataset_len = len(val_loader.dataset)

        res = 128
        n_img_channels = 3
        all_val_pred_lower_bounds = torch.zeros((min(dataset_len, len(val_loader) * val_batch_size), n_img_channels, res, res), device=device)
        all_val_pred_upper_bounds = torch.zeros((min(dataset_len, len(val_loader) * val_batch_size), n_img_channels, res, res), device=device)
        all_val_gt_samples = torch.zeros((min(dataset_len, len(val_loader) * val_batch_size), n_img_channels, res, res), device=device)
        all_val_maskes = torch.zeros((min(dataset_len, len(val_loader) * val_batch_size), n_img_channels, res, res), device=device)
        all_val_partial_gt_images = torch.zeros((min(dataset_len, len(val_loader) * val_batch_size), n_img_channels, res, res))
        all_val_masked_samples = torch.zeros((min(dataset_len, len(val_loader) * val_batch_size), n_img_channels, res, res))
        # endregion

        for val_idx, val_data in enumerate(val_loader):
            val_masked_image = val_data["mask_image"].to(device)
            val_gt_image = val_data["gt_image"].to(device)
            val_mask = val_data["mask"].to(device)

            val_pred_lower_bound, val_pred_upper_bound, _ = diffusion_with_bounds(val_masked_image)

            val_pred_lower_bound = (torch.zeros_like(val_pred_lower_bound) * (1. - val_mask) + val_mask * val_pred_lower_bound)
            val_pred_upper_bound = (torch.zeros_like(val_pred_upper_bound) * (1. - val_mask) + val_mask * val_pred_upper_bound)
            partial_gt = (torch.zeros_like(val_pred_upper_bound) * (1. - val_mask) + val_mask * val_gt_image)

            # region HOLD ON TO THE SAMPLE
            start_idx = val_idx * val_batch_size
            end_idx = min(dataset_len, ((val_idx + 1) * val_batch_size))
            all_val_pred_lower_bounds[start_idx:end_idx] = val_pred_lower_bound
            all_val_pred_upper_bounds[start_idx:end_idx] = val_pred_upper_bound
            all_val_gt_samples[start_idx:end_idx] = val_gt_image
            all_val_maskes[start_idx:end_idx] = val_data['mask']
            all_val_partial_gt_images[start_idx:end_idx] = partial_gt
            all_val_masked_samples[start_idx:end_idx] = val_data['mask_image']
            #endregion

        # region NORM TO [0,1]
        # Transform all tensors to [0,1] for calibration and metric calculation
        all_val_gt_samples = Metrics.normalize_tensor(all_val_gt_samples)
        all_val_pred_lower_bounds = Metrics.normalize_tensor(all_val_pred_lower_bounds)
        all_val_pred_upper_bounds = Metrics.normalize_tensor(all_val_pred_upper_bounds)
        all_val_partial_gt_images = Metrics.normalize_tensor(all_val_partial_gt_images)
        all_val_masked_samples = Metrics.normalize_tensor(all_val_masked_samples)
        # endregion



        pred_test_calibrated_l = (all_val_pred_lower_bounds / diffusion_with_bounds.pred_lambda_hat).clamp(0., 1.)
        pred_test_calibrated_u = (all_val_pred_upper_bounds * diffusion_with_bounds.pred_lambda_hat).clamp(0., 1.)
        calibrated_pred_risks_losses, calibrated_pred_sizes_mean, calibrated_pred_sizes_median, calibrated_pred_stratified_risks = get_rcps_metrics(pred_test_calibrated_l, pred_test_calibrated_u, all_val_gt_samples, all_val_maskes)

        # region LOG TO WANDB
        if wandb_logger:
            image_grid = vis_util.create_image_grid_inpainting(pred_test_calibrated_l.detach().cpu() * 2 - 1,
                                           pred_test_calibrated_u.detach().cpu() * 2 - 1,
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
    dist_util.sync_params(diffusion_with_bounds.parameters())
    MixedPrecisionTrainer(model=diffusion_with_bounds, use_fp16=True, fp16_scale_growth=1e-3, )


    test_set = InpaintDataset(data_root=os.path.join(args.dataset_root,"test","ground_truth"), mask_config={"mask_mode": "center"}, image_size=[128, 128])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=1, pin_memory=True)

    run_test(diffusion_with_bounds, wandb_logger, dist_util.dev(),  0, test_loader)




def create_argparser():
    defaults = dict(clip_denoised=True, num_samples=10000, batch_size=16, use_ddim=False, model_path="",)
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--prediction_time_step', type=int, default=15)
    parser.add_argument('--options_path', default=None)
    parser.add_argument('--dataset_root', type=str, default="../datasets/celebahq_256/")
    add_dict_to_argparser(parser, defaults)
    return parser
if __name__ == "__main__":
    main()
