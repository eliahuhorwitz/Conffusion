import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn

from .pinball import PinballLoss

logger = logging.getLogger('base')


class Conffusion(nn.Module):
    def __init__(self, baseModel, opt, device, load_finetuned=False):
        super(Conffusion, self).__init__()
        self.opt = opt
        self.device = device
        self.baseModel = baseModel.netG
        self.prediction_time_step = opt['train']['prediction_time_step']

        if opt['train']['finetune_loss'] == 'l2':
            self.criterion = nn.MSELoss(reduction='mean').to(self.device)
        elif opt['train']['finetune_loss'] == 'quantile_regression':
            self.q_lo_loss = PinballLoss(quantile=0.05)
            self.q_hi_loss = PinballLoss(quantile=0.95)
        self.log_dict = OrderedDict()
        self.lambda_hat = 0
        if load_finetuned:
            self.load_finetuned_network()


    def forward(self, low_res_images):
        batch_size = low_res_images.shape[0]
        noise_level = torch.FloatTensor([self.baseModel.sqrt_alphas_cumprod_prev[self.prediction_time_step + 1]]).repeat(batch_size, 1).to(self.device)
        predicted_l, predicted_u = self.baseModel.denoise_fn(torch.cat([low_res_images, torch.zeros_like(low_res_images)], dim=1), noise_level, out_upper_lower=True)

        predicted_l = self.baseModel.predict_start_from_noise(low_res_images, self.prediction_time_step, predicted_l)
        predicted_u = self.baseModel.predict_start_from_noise(low_res_images, self.prediction_time_step, predicted_u)

        predicted_l.clamp_(-1., 1.)
        predicted_u.clamp_(-1., 1.)

        return predicted_l, predicted_u


    def bounds_regression_loss_fn(self, pred_l, pred_u, gt_l, gt_u):
        lower_loss = self.criterion(pred_l, gt_l)
        upper_loss = self.criterion(pred_u, gt_u)
        loss = lower_loss + upper_loss
        return loss

    def quantile_regression_loss_fn(self, pred_l, pred_u, gt_hr):
        lower_loss = self.q_lo_loss(pred_l, gt_hr)
        upper_loss = self.q_hi_loss(pred_u, gt_hr)
        loss = lower_loss + upper_loss
        return loss

    def get_current_log(self):
        return self.log_dict

    def save_best_network(self, epoch, iter_step, optimizer, pred_lambda_hat):
        gen_path = os.path.join(self.opt['path']['checkpoint'], 'best_network_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(self.opt['path']['checkpoint'], 'best_network_opt.pth'.format(iter_step, epoch))
        # gen
        state_dict = self.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step, 'pred_lambda_hat': pred_lambda_hat, 'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = optimizer.state_dict()
        torch.save(opt_state, opt_path)


    def load_finetuned_network(self):
        load_path = self.opt['path']['bounds_resume_state']
        if load_path is not None:
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            network = self
            if isinstance(self, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']

            saved_opt = torch.load(opt_path)
            self.pred_lambda_hat = saved_opt['pred_lambda_hat']
