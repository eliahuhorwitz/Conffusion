import numpy as np
import util as Util


def create_image_grid(pred_l, pred_u, partial_gt, masked_input, gt_sample, n_rows=5):
    try:
        image_rows = []
        for i in range(n_rows):
            image_row = create_image_row(pred_l=pred_l[i].detach(), pred_u=pred_u[i].detach(), partial_gt=partial_gt[i],
                                         masked_input=masked_input[i], gt_sample=gt_sample[i])
            image_rows.append(image_row)
        image_grid = np.concatenate(image_rows, axis=0)
        return image_grid
    except:
        print("Error while creating image grid!")
        return None

def create_image_row(pred_l, pred_u, partial_gt, masked_input, gt_sample):
    pred_lower_bound_img = Util.tensor2img(pred_l)
    pred_upper_bound_img = Util.tensor2img(pred_u)
    masked_sample_img = Util.tensor2img(masked_input)
    gt_sample_img = Util.tensor2img(gt_sample)
    partial_gt_img = Util.tensor2img(partial_gt)
    image_row = np.concatenate((pred_lower_bound_img, pred_upper_bound_img, partial_gt_img, masked_sample_img, gt_sample_img), axis=1)
    return image_row

def log_train(diffusion_with_bounds, wandb_logger, pred_l, pred_u, partial_gt, train_data):
    logs = diffusion_with_bounds.get_current_log()
    image_to_log = create_image_row(pred_l=pred_l[0].detach().cpu(), pred_u=pred_u[0].detach().cpu(), partial_gt=partial_gt[0].detach().cpu(),
                                    masked_input=train_data['cond_image'][0], gt_sample=train_data['gt_image'][0])

    if wandb_logger:
        wandb_logger.log_image("Finetune/Images", image_to_log, caption="Pred L, Pred U, GT, Masked Input, Full GT", commit=False)
        wandb_logger.log_metrics(logs)