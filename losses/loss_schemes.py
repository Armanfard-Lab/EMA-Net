#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F

def model_fit(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    if task_type == 'depth' or task_type == 'normals':
        binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == 'semseg' or task_type == 'human_parts':
        # semantic loss: depth-wise cross entropy
        if task_type == 'human_parts' and (x_output == -1).all(): # check if batch doesn't have parts
            loss = torch.tensor(0.0, requires_grad=True, device=device)
        else:
            loss = F.nll_loss(x_pred, x_output.long(), ignore_index=-1)

    if task_type == 'depth':
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

    if task_type == 'normals':
        # normal loss: dot product
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)
    
    if task_type == 'sal':
        mask = (x_output != -1).to(device)
        masked_label = torch.masked_select(x_output, mask)
        assert torch.max(masked_label) < 2  # binary
        num_labels_neg = torch.sum(1.0 - masked_label)
        num_total = torch.numel(masked_label)
        w_pos = num_labels_neg / num_total
        class_weight = torch.stack((1. - w_pos, w_pos), dim=0)
        loss = F.cross_entropy(
            x_pred, x_output.long(), weight=class_weight, ignore_index=-1)

    return loss

def finalize(x, task):
    if task == 'semseg' or task == 'human_parts':
        x = F.log_softmax(x, dim=1)
    elif task =='normals':
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
    elif task =='depth' or task == 'sal':
        x = x
    else:
        raise NotImplementedError
    return x

class SingleTaskLoss(nn.Module):
    def __init__(self, tasks):
        super(SingleTaskLoss, self).__init__()
        self.tasks = tasks
    
    def forward(self, pred, gt):
        pred = {task: finalize(pred[task], task) for task in self.tasks}

        preds = {}
        preds['initial'] = {task: pred[task] for task in self.tasks}
        preds['final'] = preds['initial']
        
        losses = {}
        losses['initial'] = {task: model_fit(pred[task], gt[task], task) for task in self.tasks}
        losses['final'] = losses['initial']
        return losses, preds


class MultiTaskLoss(nn.Module):
    def __init__(self, tasks: list, loss_weighting: dict):
        super(MultiTaskLoss, self).__init__()
        self.tasks = tasks
        self.loss_weighting = loss_weighting
    
    def forward(self, pred, gt):
        pred = {task: finalize(pred[task], task) for task in self.tasks}

        preds = {}
        preds['initial'] = {task: pred[task] for task in self.tasks}
        preds['final'] = preds['initial']

        losses = {}
        losses['initial'] = {task: model_fit(pred[task], gt[task], task) for task in self.tasks}
        losses['final'] = losses['initial']
        losses['total'] = self.loss_weighting(losses)
        return losses, preds


class PADNetLoss(nn.Module):
    def __init__(self, tasks: list, auxilary_tasks: list, loss_weighting: dict):
        super(PADNetLoss, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.loss_weighting = loss_weighting
    
    def forward(self, pred, gt):
        preds = {'initial': {}, 'final': {}}
        losses = {'initial': {}, 'final': {}}

        # Losses initial task predictions (deepsup)
        for task in self.auxilary_tasks:
            pred_, gt_= finalize(pred[f'initial_{task}'], task), gt[task]
            loss_ = model_fit(pred_, gt_, task)
            preds['initial'][task] = pred_
            losses['initial'][task] = loss_


        # Losses at output  
        for task in self.tasks:
            pred_, gt_ = finalize(pred[task], task), gt[task]
            loss_ = model_fit(pred_, gt_, task)
            preds['final'][task] = pred_
            losses['final'][task] = loss_

        losses['total'] = self.loss_weighting(losses)

        return losses, preds

class PAPNetLoss(nn.Module):
    def __init__(self, tasks: list, auxilary_tasks: list, loss_weighting: dict):
        super(PAPNetLoss, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks

        self.loss_weighting = loss_weighting
    
    # TODO: Add pair-wise loss
    def forward(self, pred, gt):
        preds = {'initial': {}, 'final': {}}
        losses = {'initial': {}, 'final': {}}

        # Losses initial task predictions (deepsup)
        for task in self.auxilary_tasks:
            pred_, gt_= finalize(pred[f'initial_{task}'], task), gt[task]
            loss_ = model_fit(pred_, gt_, task)
            preds['initial'][task] = pred_
            losses['initial'][task] = loss_


        # Losses at output  
        for task in self.tasks:
            pred_, gt_ = finalize(pred[task], task), gt[task]
            loss_ = model_fit(pred_, gt_, task)
            preds['final'][task] = pred_
            losses['final'][task] = loss_

        losses['total'] = self.loss_weighting(losses)

        return losses, preds
    
class EMANetLoss(nn.Module):
    def __init__(self, tasks: list, auxilary_tasks: list, loss_weighting: dict):
        super(EMANetLoss, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.loss_weighting = loss_weighting
    
    def forward(self, pred, gt):
        preds = {'initial': {}, 'final': {}}
        losses = {'initial': {}, 'final': {}}

        # Losses initial task predictions (deepsup)
        for task in self.auxilary_tasks:
            pred_, gt_= finalize(pred[f'initial_{task}'], task), gt[task]
            loss_ =model_fit(pred_, gt_, task)
            preds['initial'][task] = pred_
            losses['initial'][task] = loss_

        # Losses at output  
        for task in self.tasks:
            pred_, gt_ = finalize(pred[task], task), gt[task]
            loss_ = model_fit(pred_, gt_, task)
            preds['final'][task] = pred_
            losses['final'][task] = loss_

        losses['total'] = self.loss_weighting(losses)

        return losses, preds

class MTINetLoss(nn.Module):
    def __init__(self, tasks: list, auxilary_tasks: list, loss_weighting: dict):
        super(MTINetLoss, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.loss_weighting = loss_weighting
    
    def forward(self, pred, gt):
        preds = {'initial': {}, 'final': {}}
        losses = {'initial': {t: 0. for t in self.auxilary_tasks}}

        img_size = gt[self.tasks[0]].size()[-2:]
        
        # Losses initial task predictions at multiple scales (deepsup)
        for scale in range(4):
            pred_scale = pred['deep_supervision']['scale_%s' %(scale)]
            pred_scale = {t: F.interpolate(pred_scale[t], img_size, mode='bilinear') for t in self.auxilary_tasks}
            pred_scale = {t: finalize(pred_scale[t], t) for t in self.auxilary_tasks}
            if scale == 0:
                preds['initial'] = pred_scale
            losses_scale = {t: model_fit(pred_scale[t], gt[t], t) for t in self.auxilary_tasks}
            for k, v in losses_scale.items():
                losses['initial'][k] += v/4

        # Losses at output
        preds['final'] = {task: finalize(pred[task], task) for task in self.tasks}
        losses['final'] = {task: model_fit(preds['final'][task], gt[task], task) for task in self.tasks}

        losses['total'] = self.loss_weighting(losses)

        return losses, preds