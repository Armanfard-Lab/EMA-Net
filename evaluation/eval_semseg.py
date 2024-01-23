# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import warnings
import cv2
import os.path
import glob
import json
import numpy as np
import torch
from PIL import Image
import pdb

class SemsegMeter(object):
    def __init__(self, num_classes):
            self.num_classes = num_classes
            self.mat = None

    @torch.no_grad()
    def update(self, pred, target):
        pred = pred.argmax(1).flatten()
        target = target.flatten()
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_score(self):
        eval_result = dict()

        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))

        eval_result['mIoU'] = torch.nanmean(iu).item()
        eval_result['pix_acc'] = acc.item()

        return eval_result