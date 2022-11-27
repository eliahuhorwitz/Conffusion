from collections import OrderedDict

import blobfile as bf
import torch
import torch.distributed as dist
import torch.nn as nn
from guided_diffusion import logger

from .pinball import PinballLoss


class Conffusion(nn.Module):
    def __init__(self, diffusion, model, device, load_finetuned=False, prediction_time_step=15, options_path=None):
        super(Conffusion, self).__init__()
        self.options_path = options_path
        self.device = device
        self.diffusion = diffusion
        self.model = model
        self.prediction_time_step = prediction_time_step
        self.q_lo_loss = PinballLoss(quantile=0.05)
        self.q_hi_loss = PinballLoss(quantile=0.95)
        self.log_dict = OrderedDict()
        self.lambda_hat = 0
        if load_finetuned:
            self.load_finetuned_network()

    def forward(self, input_img):
        batch_size = input_img.shape[0]
        t = torch.tensor([self.prediction_time_step] * batch_size, device=self.device)
        classes = torch.randint(low=0, high=1000, size=(batch_size,), device=self.device) # Only for class conditional model


        noisy_input_img = self.diffusion.q_sample(input_img, t)

        predicted_l_temp, predicted_u_temp = self.model(noisy_input_img, t, y=classes, out_upper_lower=True)

        predicted_l_out, predicted_l_var_values = torch.split(predicted_l_temp, 3, dim=1)
        predicted_u_out, predicted_u_var_values = torch.split(predicted_u_temp, 3, dim=1)


        predicted_l = self.diffusion._predict_xstart_from_eps(noisy_input_img, t, predicted_l_out)
        predicted_u = self.diffusion._predict_xstart_from_eps(noisy_input_img, t, predicted_u_out)

        predicted_l.clamp_(-1., 1.)
        predicted_u.clamp_(-1., 1.)

        return predicted_l, predicted_u, noisy_input_img

    def quantile_regression_loss_fn(self, pred_l, pred_u, gt_hr):
        lower_loss = self.q_lo_loss(pred_l, gt_hr)
        upper_loss = self.q_hi_loss(pred_u, gt_hr)
        loss = lower_loss + upper_loss
        return loss

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.model.convert_to_fp16()

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float16.
        """
        self.model.convert_to_fp32()

    def get_current_log(self):
        return self.log_dict

    def save_best_network(self, mp_trainer, pred_lambda_hat, epoch, iter_step, optimizer):

        state_dict = mp_trainer.master_params_to_state_dict(mp_trainer.master_params)
        if dist.get_rank() == 0:
            filename = f"best_network.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                torch.save(state_dict, f)

        if dist.get_rank() == 0:
            options = {'epoch': epoch, 'iter': iter_step, 'pred_lambda_hat': pred_lambda_hat}
            with bf.BlobFile(bf.join(get_blob_logdir(), f"best_options.pt"), "wb", ) as f:
                torch.save(options, f)
        print(f"Saved best network at step {iter_step}...")
        dist.barrier()



    def load_finetuned_network(self):
        saved_opt = torch.load(self.options_path)
        self.pred_lambda_hat = saved_opt['pred_lambda_hat']


def get_blob_logdir():
    return logger.get_dir()