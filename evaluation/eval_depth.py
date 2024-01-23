# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import numpy as np
import torch

class DepthMeter(object):
    def __init__(self):
        self.total_rmses = 0.0
        self.total_log_rmses = 0.0
        self.n_valid = 0.0

        self.rel_err = 0.0
        self.abs_err = 0.0

    @torch.no_grad()
    def update(self, pred, gt):
        device = pred.device
        
        # Determine valid mask
        mask = (torch.sum(gt, dim=1) != 0).unsqueeze(1).to(device)
        self.n_valid += mask.float().sum().item() # Valid pixels per image
        
        pred_true = pred.masked_select(mask)
        gt_true = gt.masked_select(mask)

        pred_true[pred_true<=0] = 1e-9

        # Per pixel rmse and log-rmse.
        log_rmse_tmp = torch.pow(torch.log(gt_true) - torch.log(pred_true), 2)
        self.total_log_rmses += log_rmse_tmp.sum().item()

        rmse_tmp = torch.pow(gt_true - pred_true, 2)
        self.total_rmses += rmse_tmp.sum().item()

        # rel
        self.rel_err += (torch.abs(gt_true - pred_true) / gt_true).sum().item()
        # abs
        self.abs_err += (torch.abs(gt_true - pred_true)).sum().item()

    def get_score(self):
        eval_result = dict()
        eval_result['rmse'] = np.sqrt(self.total_rmses / self.n_valid)
        eval_result['log_rmse'] = np.sqrt(self.total_log_rmses / self.n_valid)
        eval_result['rel'] = self.rel_err / self.n_valid
        eval_result['abs'] = self.abs_err / self.n_valid

        return eval_result