import numpy as np
import torch

from .calibration import fraction_missed_loss


def get_quantile_sample_bounds(sample_variations, l_q=0.05, u_q=0.95):
    """
    sample_variations: shape is [n_soch_samples, n_img_channels, sr_res, sr_res]
    """
    upper_bound = torch.quantile(sample_variations, u_q, dim=0)
    lower_bound = torch.quantile(sample_variations, l_q, dim=0)
    return upper_bound, lower_bound





def get_rcps_metrics(lower_bound, upper_bound, ground_truth):
  risks_losses = []
  sizes = []

  # risk_vis = fraction_missed_loss(lower_bound, upper_bound, ground_truth, avg_channels=False)[1]
  for idx in range(upper_bound.shape[0]):
    risks_losses = risks_losses + [fraction_missed_loss(lower_bound[idx], upper_bound[idx], ground_truth[idx], avg_channels=False)[0].unsqueeze(dim=0),]
    sizes = sizes + [(upper_bound[idx]-lower_bound[idx]).mean(dim=(1,2)).unsqueeze(dim=0),]


  risks_losses = torch.cat(risks_losses, dim=0)
  sizes = torch.cat(sizes, dim=0)
  sizes = sizes + torch.rand(size=sizes.shape,device=sizes.device) * 1e-6

  size_bins = torch.tensor([0, torch.quantile(sizes, 0.25), torch.quantile(sizes, 0.5), torch.quantile(sizes, 0.75)], device=sizes.device)
  buckets = torch.bucketize(sizes, size_bins)-1
  stratified_risks = torch.tensor([torch.nan_to_num(risks_losses[buckets == bucket].mean()) for bucket in range(size_bins.shape[0])])

  return risks_losses.mean(), sizes.mean(), sizes.median(), stratified_risks#, risk_vis