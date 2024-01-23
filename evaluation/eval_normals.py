# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import numpy as np
import torch

class NormalsMeter(object):
    def __init__(self):
        self.sum_deg_diff = 0
        self.sum_deg_diff_11_25 = 0
        self.sum_deg_diff_22_5 = 0
        self.sum_deg_diff_30 = 0
        self.total = 0

    @torch.no_grad()
    def update(self, pred, gt):
        mask = (torch.sum(gt, dim=1) != 0)

        error = torch.acos(torch.clamp(torch.sum(pred * gt, 1).masked_select(mask), -1, 1)).detach().cpu().numpy()
        error = torch.from_numpy(np.degrees(error))

        self.sum_deg_diff += torch.sum(error).cpu().item()
        self.sum_deg_diff_11_25 += torch.sum(error < 11.25)
        self.sum_deg_diff_22_5 += torch.sum(error < 22.5)
        self.sum_deg_diff_30 += torch.sum(error < 30)
        self.total += error.numel()

    def get_score(self):
        eval_result = dict()
        eval_result['mean'] = self.sum_deg_diff / self.total
        eval_result['11.25'] = self.sum_deg_diff_11_25 / self.total
        eval_result['22.5'] = self.sum_deg_diff_22_5 / self.total
        eval_result['30'] = self.sum_deg_diff_30 / self.total

        return eval_result