import argparse

import conffusion.metrics as Metrics
import sr_dataset as Data
import torch
import wandb
from conffusion import vis_util
from conffusion.add_bounds import Conffusion
from conffusion.calibration import calibrate_model
from conffusion.eval_utils import get_rcps_metrics
from conffusion.wandb_logger import WandbLogger
from guided_diffusion import dist_util, logger, script_util
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.script_util import (add_dict_to_argparser, args_to_dict,
                                          create_model_and_diffusion,
                                          model_and_diffusion_defaults)


def run_validation(diffusion_with_bounds, wandb_logger, device, val_step, val_loader):
    print("Starting Validation...")
    with torch.no_grad():
        # region INIT VARS
        val_loss = 0
        val_batch_size = val_loader.batch_size
        dataset_len = len(val_loader.dataset)

        sr_res = 128
        n_img_channels = 3
        all_val_pred_lower_bounds = torch.zeros((min(dataset_len, len(val_loader) * val_batch_size), n_img_channels, sr_res, sr_res), device=device)
        all_val_pred_upper_bounds = torch.zeros((min(dataset_len, len(val_loader) * val_batch_size), n_img_channels, sr_res, sr_res), device=device)
        all_val_gt_samples = torch.zeros((min(dataset_len, len(val_loader) * val_batch_size), n_img_channels, sr_res, sr_res), device=device)
        all_val_lr_samples = torch.zeros((min(dataset_len, len(val_loader) * val_batch_size), n_img_channels, sr_res, sr_res), device=device)
        # endregion

        for val_idx, val_data in enumerate(val_loader):
            val_low_res_image = val_data["SR"].to(device)
            val_gt_image = val_data['HR'].to(device)

            val_pred_lower_bound, val_pred_upper_bound, noisy_input_img = diffusion_with_bounds(val_low_res_image)


            val_bounds_loss = diffusion_with_bounds.quantile_regression_loss_fn(val_pred_lower_bound, val_pred_upper_bound, val_gt_image)
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



        pred_val_lambda_hat, pred_val_calibrated_l, pred_val_calibrated_u = calibrate_model(all_val_pred_lower_bounds, all_val_pred_upper_bounds, all_val_gt_samples)
        calibrated_pred_risks_losses, calibrated_pred_sizes_mean, calibrated_pred_sizes_median, calibrated_pred_stratified_risks = get_rcps_metrics(pred_val_calibrated_l, pred_val_calibrated_u, all_val_gt_samples)

        # region LOG TO WANDB
        if wandb_logger:
            image_grid = vis_util.create_image_grid_sr(pred_val_calibrated_l.detach().cpu() * 2 - 1,
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

def run_training(diffusion_with_bounds, wandb_logger, device, optimizer, mp_trainer, train_loader, val_loader):
    val_step = 0
    current_step = 0
    current_epoch = 0
    n_iter = 15000

    best_interval_size = float('inf')

    while current_step < n_iter:
        current_epoch += 1

        for train_idx, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > n_iter:
                break
            low_res_image = train_data["SR"].to(device)
            gt_image = train_data['HR'].to(device)

            pred_lower_bound, pred_upper_bound, noisy_input_img = diffusion_with_bounds(low_res_image)
            mp_trainer.zero_grad()
            bounds_loss = diffusion_with_bounds.quantile_regression_loss_fn(pred_lower_bound, pred_upper_bound, gt_image)


            diffusion_with_bounds.log_dict['Finetune/Loss'] = bounds_loss.item()
            mp_trainer.backward(bounds_loss)

            _ = mp_trainer.optimize(optimizer)

            if current_step % 200 == 0 or current_step - 1 == 0:
                vis_util.log_train_sr(diffusion_with_bounds, wandb_logger, pred_lower_bound, pred_upper_bound, train_data)

            # region VALIDATION
            if current_step % 200 == 0 or current_step - 1 == 0:
                calibrated_pred_risks_losses, calibrated_pred_sizes, pred_val_lambda_hat = run_validation(diffusion_with_bounds, wandb_logger, device, val_step, val_loader)
                val_step += 1
                if calibrated_pred_risks_losses < 0.1 and calibrated_pred_sizes < best_interval_size:
                    logger.info(f'Saving best model and training states, interval size is {calibrated_pred_sizes}, lambda_hat is {pred_val_lambda_hat}')

                    diffusion_with_bounds.save_best_network(mp_trainer, pred_val_lambda_hat, current_epoch, current_step, optimizer)
                    best_interval_size = calibrated_pred_sizes
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
    if args.model_path != "":
        model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"), strict=False)



    # Initialize Prediction Heads
    with torch.no_grad():
        model.lower[0].weight.copy_(model.out[0].weight)
        model.lower[0].bias.copy_(model.out[0].bias)
        model.lower[2].weight.copy_(model.out[2].weight)
        model.lower[2].bias.copy_(model.out[2].bias)

        model.upper[0].weight.copy_(model.out[0].weight)
        model.upper[0].bias.copy_(model.out[0].bias)
        model.upper[2].weight.copy_(model.out[2].weight)
        model.upper[2].bias.copy_(model.out[2].bias)
    model.to(dist_util.dev())


    diffusion_with_bounds = Conffusion(diffusion, model, dist_util.dev(), load_finetuned=False, prediction_time_step=args.prediction_time_step).to(dist_util.dev())
    diffusion_with_bounds.train()
    dist_util.sync_params(diffusion_with_bounds.parameters())

    mp_trainer = MixedPrecisionTrainer(model=diffusion_with_bounds, use_fp16=True, fp16_scale_growth=1e-3,)
    model.train()

    weight_decay = 0
    optimizer = torch.optim.AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=weight_decay)

    train_set = Data.create_dataset(args.dataset_root, "calibration")
    train_loader = Data.create_dataloader(train_set, "calibration")
    val_set = Data.create_dataset(args.dataset_root, "validation")
    val_loader = Data.create_dataloader(val_set, "validation")
    run_training(diffusion_with_bounds, wandb_logger, dist_util.dev(), optimizer,mp_trainer,  train_loader, val_loader)




def create_argparser():
    defaults = dict(clip_denoised=True, num_samples=10000, batch_size=16, use_ddim=False, model_path="",)
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--prediction_time_step', type=int, default=3)
    parser.add_argument('--dataset_root', type=str, default="../datasets/celebahq_16_128")
    add_dict_to_argparser(parser, defaults)

    return parser
if __name__ == "__main__":
    main()
