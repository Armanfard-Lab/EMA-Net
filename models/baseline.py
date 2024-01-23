import torch
import torch.nn as nn
import torch.nn.functional as F
        
class HPS(nn.Module):
    def __init__(self, opt, backbone, backbone_channels, heads) -> None:
        super(HPS, self).__init__()
        # General
        self.tasks = opt.TASKS.NAMES
        self.channels = backbone_channels
        self.out_channels = opt.TASKS.NUM_OUTPUT

        # Backbone
        self.backbone = backbone

        # Task-specific heads for initial prediction 
        self.heads = heads

    def forward(self, x):
        img_size = x.size()[-2:]
        out = {}
        
        # Backbone
        x = self.backbone(x)

        # Initial predictions for every task
        for task in self.tasks:
            y = self.heads[task](x)
            out[task] = F.interpolate(y, img_size, mode='bilinear')

        return out

class STL(nn.Module):
    def __init__(self, opt, backbones, backbone_channels, heads) -> None:
        super(STL, self).__init__()
        # General
        self.tasks = opt.TASKS.NAMES
        self.channels = backbone_channels
        self.out_channels = opt.TASKS.NUM_OUTPUT

        # Backbones for each task
        self.backbones = backbones

        # Task-specific heads for initial prediction
        self.heads = heads

    def forward(self, x):
        img_size = x.size()[-2:]
        out = {}
        
        # Backbone
        x = {task: self.backbones[task](x) for task in self.tasks}

        # Initial predictions for every task
        for task in self.tasks:
            y = self.heads[task](x[task])
            out[task] = F.interpolate(y, img_size, mode='bilinear')

        return out