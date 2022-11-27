import metrics as Metrics
import numpy as np


def create_image_grid(pred_l, pred_u, low_res, gt_sample, n_rows=5):
    try:
        image_rows = []
        for i in range(n_rows):
            image_row = create_image_row(pred_l=pred_l[i].detach(), pred_u=pred_u[i].detach(), low_res=low_res[i].detach(), gt_sample=gt_sample[i].detach())
            image_rows.append(image_row)
        image_grid = np.concatenate(image_rows, axis=0)
        return image_grid
    except:
        print("Error while creating image grid!")
        return None

def create_image_row(pred_l, pred_u, low_res, gt_sample):
    pred_lower_bound_img = Metrics.tensor2img(pred_l)
    pred_upper_bound_img = Metrics.tensor2img(pred_u)
    low_res_img = Metrics.tensor2img(low_res)
    gt_sample_img = Metrics.tensor2img(gt_sample)
    image_row = np.concatenate((pred_lower_bound_img, pred_upper_bound_img, low_res_img, gt_sample_img), axis=1)
    return image_row

def log_train(diffusion_with_bounds, wandb_logger, pred_l, pred_u, train_data):
    logs = diffusion_with_bounds.get_current_log()
    image_to_log = create_image_row(pred_l=pred_l[0].detach(), pred_u=pred_u[0].detach(), low_res=train_data['SR'][0], gt_sample=train_data['HR'][0])
    if wandb_logger:
        wandb_logger.log_image("Finetune/Images", image_to_log, caption="Pred L, Pred U, LR, GT SR", commit=False)
        wandb_logger.log_metrics(logs)
