import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d
import torch.utils.data.sampler as sampler
from torchvision.models import ResNet18_Weights, ResNet34_Weights
from models.baseline import *
from losses import *

from torch.cuda.amp import autocast
from utils import *

class MTR_Net(nn.Module):
    def __init__(self, tasks, model, criterion) -> None:
        super(MTR_Net, self).__init__()
        self.tasks = tasks
        self.model = model
        self.criterion = criterion

    def set_optimizers(self, opt):
        self.optimizer = torch.optim.Adam(list(self.model.parameters()), opt['optimizer_kwargs']['lr'], weight_decay=opt['optimizer_kwargs']['weight_decay'])

    def set_schedulers(self, opt):
        if opt['scheduler'] == 'none':
            self.scheduler = None
        elif opt['scheduler'] == 'poly':
            from torch.optim.lr_scheduler import PolynomialLR
            self.scheduler = PolynomialLR(self.optimizer, opt['epochs'], power=2, verbose=True)
        elif opt['scheduler'] == 'step':
            from torch.optim.lr_scheduler import StepLR
            self.scheduler = StepLR(self.optimizer, step_size=opt.epochs//2, gamma=0.5, verbose=True)
        elif opt['scheduler'] == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=opt.epochs, verbose=True)
        elif opt['scheduler'] == 'cosine_wr':
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=opt.epochs//3, T_mult=2, verbose=True)
        else:
            raise NotImplementedError

    def forward(self, x):
        pass
        
    def mt_step(self, x, scaler=None, train=False):
        if scaler is not None and train:
            with autocast():
                out = self.model(x['image'])
                losses, preds = self.criterion(out, x)
            
                total_loss = losses['total']
                self.optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
        else:
            out = self.model(x['image'])
            losses, preds = self.criterion(out, x)

            if train:
                    total_loss = losses['total']
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

        return losses['initial'], preds['initial'], losses['final'], preds['final']

class STR_Net(nn.Module):
    def __init__(self, tasks, model, criterion) -> None:
        super(STR_Net, self).__init__()
        self.tasks = tasks
        self.model = model
        self.criterion = criterion

    def set_optimizers(self, opt):
        self.optimizer = {}
        for task in self.tasks:
            params = list(self.model.backbones[task].parameters()) + list(self.model.heads[task].parameters())
            self.optimizer[task] = torch.optim.Adam(params, opt['optimizer_kwargs']['lr'], weight_decay=opt['optimizer_kwargs']['weight_decay'])

    def set_schedulers(self, opt):
        self.scheduler = {}
        for task in self.tasks:
            if opt['scheduler'] == 'none':
                self.scheduler[task] = None
            elif opt['scheduler'] == 'poly':
                from torch.optim.lr_scheduler import PolynomialLR
                self.scheduler[task] = PolynomialLR(self.optimizer[task], opt['epochs'], power=2, verbose=True)
            elif opt['scheduler'] == 'step':
                from torch.optim.lr_scheduler import StepLR
                self.scheduler[task] = StepLR(self.optimizer[task], step_size=opt.epochs//2, gamma=0.5, verbose=True)
            elif opt['scheduler'] == 'cosine':
                from torch.optim.lr_scheduler import CosineAnnealingLR
                self.scheduler[task] = CosineAnnealingLR(self.optimizer[task], T_max=opt.epochs, verbose=True)
            elif opt['scheduler'] == 'cosine_wr':
                from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
                self.scheduler[task] = CosineAnnealingWarmRestarts(self.optimizer[task], T_0=opt.epochs//3, T_mult=2, verbose=True)
            else:
                raise NotImplementedError

    def forward(self, x):
        pass
        
    def mt_step(self, x, scaler=None, train=False):
        out = self.model(x['image'])
        losses, preds = self.criterion(out, x)

        if train:
            for task in self.tasks:
                loss = losses['final'][task]

                self.optimizer[task].zero_grad()
                loss.backward()
                self.optimizer[task].step()

        return losses['initial'], preds['initial'], losses['final'], preds['final']