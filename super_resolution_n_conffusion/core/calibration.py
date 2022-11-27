import numpy as np
import torch
from core.bounds import HB_mu_plus


def fraction_missed_loss(lower_bound, upper_bound, ground_truth, avg_channels=True):
    misses = (lower_bound > ground_truth).float() + (upper_bound < ground_truth).float()
    misses[misses > 1.0] = 1.0
    if avg_channels:
        return misses.mean(), misses
    else:
        return misses.mean(dim=(1,2)), misses 


def get_rcps_losses_from_outputs(cal_l, cal_u, ground_truth, lam):
    risk, misses = fraction_missed_loss(cal_l / lam, cal_u * lam, ground_truth)
    return risk


def calibrate_model(cal_l, cal_u, ground_truth):
    alpha = 0.1
    delta = 0.1
    minimum_lambda = 0.9
    maximum_lambda = 1.3
    num_lambdas = 1000

    lambdas = torch.linspace(minimum_lambda, maximum_lambda, num_lambdas)
    dlambda = lambdas[1] - lambdas[0]
    lambda_hat = (lambdas[-1] + dlambda - 1e-9)

    for lam in reversed(lambdas):
      losses = get_rcps_losses_from_outputs(cal_l, cal_u, ground_truth, lam=(lam - dlambda))

      Rhat = losses
      RhatPlus = HB_mu_plus(Rhat.item(),  cal_l.shape[0] * cal_l.shape[1] * cal_l.shape[2] * cal_l.shape[3], delta)

      print(f"\rLambda: {lam:.4f}  |  Rhat: {Rhat:.4f}  |  RhatPlus: {RhatPlus:.4f}  ",end='')
      if Rhat >= alpha or RhatPlus > alpha:

        lambda_hat = lam
        print(f"Model's lambda_hat is {lambda_hat}")
        break
    return lambda_hat, (cal_l / lambda_hat).clamp(0., 1.), (cal_u * lambda_hat).clamp(0., 1.)


