#
# Authors: Dimitrios Sinodinos
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

"""
    Implementation of EMA-Net.
    https://arxiv.org/abs/2401.11124
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import Bottleneck

class InitialTaskPredictionModule(nn.Module):
    """
        Make the initial task predictions from the backbone features.
    """
    def __init__(self, out_channels, tasks, input_channels, intermediate_channels=256):
        super(InitialTaskPredictionModule, self).__init__() 
        self.tasks = tasks 
        layers = {}
        conv_out = {}
        
        for task in self.tasks:
            if input_channels != intermediate_channels:
                downsample = nn.Sequential(nn.Conv2d(input_channels, intermediate_channels, kernel_size=1,
                                                    stride=1, bias=False), nn.BatchNorm2d(intermediate_channels))
            else:
                downsample = None
            bottleneck1 = Bottleneck(input_channels, intermediate_channels//4, downsample=downsample)
            bottleneck2 = Bottleneck(intermediate_channels, intermediate_channels//4, downsample=None)
            conv_out_ = nn.Conv2d(intermediate_channels, out_channels[task], 1)
            layers[task] = nn.Sequential(bottleneck1, bottleneck2)
            conv_out[task] = conv_out_

        self.layers = nn.ModuleDict(layers)
        self.conv_out = nn.ModuleDict(conv_out)


    def forward(self, x):
        out = {}
        
        for task in self.tasks:
            out['features_%s' %(task)] = self.layers[task](x)
            out[task] = self.conv_out[task](out['features_%s' %(task)])
        
        return out 

class SpatialAttentionModule(nn.Module):
    def __init__(self, tasks, channels, feature_size, gamma):
        super(SpatialAttentionModule, self).__init__()
        self.tasks = tasks

        # Value projection for each task
        self.projection = nn.ModuleDict({task: nn.Conv2d(channels, channels, kernel_size=3, padding=1) for task in self.tasks})

        # Cross-Task Affinity Learning Modules
        self.fusion = nn.Conv2d(in_channels=len(self.tasks)*feature_size, out_channels=feature_size, kernel_size=3, stride=1, padding=1, groups=feature_size)

        # Blending parameter
        self.gamma = gamma

    def forward(self, x):
        out = {}
        M = []
        B, C, H, W = list(x.values())[0].size()

        # compute all self-attention masks
        for task in self.tasks:
            features = x[f'features_{task}'].view(B, C, H*W)
            # Normalize the features
            features = F.normalize(features, dim=1)
            # Compute the inner product to get the affinity matrix
            affinity_matrix = torch.bmm(features.transpose(1,2), features)
            # Reshape matrix to (B, H*W, H, W)
            affinity_matrix = affinity_matrix.view(B, -1, H, W)
            M.append(affinity_matrix)
        
        # channel-wise interleave concatenation
        M = torch.stack(M, dim=2).reshape(B, -1, H, W)
        M = self.fusion(M)

        # compute and apply all cross-attention masks
        for task in self.tasks:
            features = x[f'features_{task}']
            attention = M.view(B, H*W, H*W).transpose(1,2)
            value = self.projection[task](features).view(B, C, -1)
            attended_features = torch.bmm(value, attention).view(B, -1, H, W)
            x_out = self.gamma*attended_features + (1-self.gamma)*features
            out[f'features_{task}'] = x_out
        
        return out


class AffinityLearningModule(nn.Module):
    def __init__(self, auxilary_tasks, in_channels, out_channels, feature_size, gamma):
        super(AffinityLearningModule, self).__init__()
        self.tasks = auxilary_tasks

        self.conv_in = nn.ModuleDict({task: nn.Conv2d(in_channels, out_channels, 1) for task in self.tasks})

        # multitask spatial attention module
        self.spatial_att = SpatialAttentionModule(self.tasks, out_channels, feature_size, gamma)

        # channel reduction layers for outgoing features
        conv_out = {}
        for task in self.tasks:
            out = nn.Sequential(
                nn.ConvTranspose2d(out_channels, out_channels, 2, 2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                # nn.Dropout2d(0.1),
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
            )
            conv_out[task] = out
        self.conv_out = nn.ModuleDict(conv_out)


    def forward(self, x):
        out = {}
        for task in self.tasks:
            x[f'features_{task}'] = self.conv_in[task](x[f'features_{task}'])
        x = self.spatial_att(x)
        for task in self.tasks:
            out[f'features_{task}'] = self.conv_out[task](x[f'features_{task}'])

        return out

class EMANet(nn.Module):
    def __init__(self, opt, backbone, backbone_channels, backbone_dims):
        super(EMANet, self).__init__()
        # General
        self.tasks = opt.TASKS.NAMES
        self.auxilary_tasks = opt.TASKS.NAMES
        self.channels = backbone_channels
        self.embedding_size = torch.tensor(backbone_dims, dtype=torch.int32).prod().item()
        self.out_channels = opt.TASKS.NUM_OUTPUT
        self.gamma = opt['model_kwargs']['gamma']
        intermediate_channels = 128

        # Backbone
        self.backbone = backbone

        # Task-specific heads for initial prediction 
        self.initial_task_prediction_heads = InitialTaskPredictionModule(self.out_channels, self.auxilary_tasks, self.channels)

        # Cross-task propagation
        self.affinity_learning_module = AffinityLearningModule(self.auxilary_tasks, 256, intermediate_channels, self.embedding_size, self.gamma)

        # Task-specific heads for final prediction
        heads = {}
        for task in self.tasks:
            bottleneck1 = Bottleneck(intermediate_channels, intermediate_channels//4, downsample=None)
            bottleneck2 = Bottleneck(intermediate_channels, intermediate_channels//4, downsample=None)
            conv_out_ = nn.Conv2d(intermediate_channels, self.out_channels[task], 1)
            heads[task] = nn.Sequential(bottleneck1, bottleneck2, conv_out_)

        self.heads = nn.ModuleDict(heads)
    

    def forward(self, x):
        img_size = x.size()[-2:]
        out = {}
        
        # Backbone
        x = self.backbone(x)

        # Initial predictions for every task including auxilary tasks
        x = self.initial_task_prediction_heads(x)
        for task in self.auxilary_tasks:
            out['initial_%s' %(task)] = F.interpolate(x[task], img_size, mode='bilinear')

        # Affinty learning
        x = self.affinity_learning_module(x)

        # Make final prediction with task-specific heads
        for task in self.tasks:
            out[task] = F.interpolate(self.heads[task](x[f'features_{task}']), img_size, mode='bilinear')

        return out