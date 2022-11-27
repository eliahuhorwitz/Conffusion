import torch
from conffusion.calibration_masked import fraction_missed_loss


def get_rcps_metrics(lower_bound, upper_bound, ground_truth, masks):
  risks_losses = []
  sizes = []

  for idx in range(upper_bound.shape[0]):
    risks_losses = risks_losses + [fraction_missed_loss(lower_bound[idx], upper_bound[idx], ground_truth[idx],masks[idx],only_avg_masked=True, avg_channels=False).unsqueeze(dim=0),]
    mask_indices = torch.argwhere(masks[idx] == 1.)
    sizes = sizes + [(upper_bound[idx] - lower_bound[idx])[mask_indices[:, 0], mask_indices[:, 1], mask_indices[:, 2]].mean().unsqueeze(dim=0), ]

  risks_losses = torch.cat(risks_losses, dim=0)
  sizes = torch.cat(sizes, dim=0)
  sizes = sizes + torch.rand(size=sizes.shape,device=sizes.device) * 1e-6

  size_bins = torch.tensor([0, torch.quantile(sizes, 0.25), torch.quantile(sizes, 0.5), torch.quantile(sizes, 0.75)], device=sizes.device)
  buckets = torch.bucketize(sizes, size_bins)-1
  stratified_risks = torch.tensor([torch.nan_to_num(risks_losses[buckets == bucket].mean()) for bucket in range(size_bins.shape[0])])

  return risks_losses.mean(), sizes.mean(), sizes.median(), stratified_risks