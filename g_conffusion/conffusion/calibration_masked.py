import numpy as np
import torch
from conffusion.bounds import HB_mu_plus


def fraction_missed_loss(lower_bound, upper_bound, ground_truth, masks, only_avg_masked=True, avg_channels=True):
    misses = (lower_bound > ground_truth).float() + (upper_bound < ground_truth).float()
    misses[misses > 1.0] = 1.0
    if only_avg_masked:
        if len(misses.shape) == 3:
            mask_indices = torch.argwhere(masks == 1.)
            unmasked_misses = misses[mask_indices[:, 0], mask_indices[:, 1], mask_indices[:, 2]]
            return unmasked_misses.mean()
        elif len(misses.shape) == 4:
            mask_indices = torch.argwhere(masks == 1.)
            unmasked_misses = misses[mask_indices[:,0], mask_indices[:,1], mask_indices[:,2], mask_indices[:,3]]
            return unmasked_misses.mean()
        else:
            raise NotImplementedError('fraction_missed_loss does not support this shape')
    if avg_channels:
        return misses.mean()
    else:
        return misses.mean(dim=(1,2))


def get_rcps_losses_from_outputs(cal_l, cal_u, ground_truth, lam, masks):
    risk = fraction_missed_loss(cal_l / lam, cal_u * lam, ground_truth, masks, only_avg_masked=True)
    return risk


def calibrate_model(cal_l, cal_u, ground_truth, masks):
    alpha = 0.1
    delta = 0.1
    minimum_lambda = 0.9
    maximum_lambda = 2.5
    num_lambdas = 1000

    lambdas = torch.linspace(minimum_lambda, maximum_lambda, num_lambdas)
    dlambda = lambdas[1] - lambdas[0]
    lambda_hat = (lambdas[-1] + dlambda - 1e-9)

    mask_indices_len = torch.argwhere(masks == 1.).shape[0]
    for lam in reversed(lambdas):
      losses = get_rcps_losses_from_outputs(cal_l, cal_u, ground_truth, lam=(lam - dlambda), masks=masks)

      Rhat = losses
      RhatPlus = HB_mu_plus(Rhat.item(), mask_indices_len, delta)
      print(f"\rLambda: {lam:.4f}  |  Rhat: {Rhat:.4f}  |  RhatPlus: {RhatPlus:.4f}  ",end='')
      if Rhat >= alpha or RhatPlus > alpha:

        lambda_hat = lam
        print(f"Model's lambda_hat is {lambda_hat}")
        break
    return lambda_hat, (cal_l / lambda_hat).clamp(0., 1.), (cal_u * lambda_hat).clamp(0., 1.)


