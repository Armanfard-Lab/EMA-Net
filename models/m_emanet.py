#
# Authors: Dimitrios Sinodinos, Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

""" 
    Multiscale version of EMA-Net that borrows some elemnets from 
    the MTI-Net implementation based on HRNet backbone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import BasicBlock, Bottleneck
from models.layers import SEBlock
from models.hrnet import HighResolutionFuse


class InitialTaskPredictionModule(nn.Module):
    """ Module to make the inital task predictions """
    def __init__(self, opt, auxilary_tasks, input_channels, task_channels):
        super(InitialTaskPredictionModule, self).__init__()        
        self.auxilary_tasks = auxilary_tasks

        # Per task feature refinement + decoding
        if input_channels == task_channels:
            channels = input_channels
            self.refinement = nn.ModuleDict({task: nn.Sequential(BasicBlock(channels, channels), BasicBlock(channels, channels)) for task in self.auxilary_tasks})
        
        else:
            refinement = {}
            for t in auxilary_tasks:
                downsample = nn.Sequential(nn.Conv2d(input_channels, task_channels, 1, bias=False), 
                                nn.BatchNorm2d(task_channels))
                refinement[t] = nn.Sequential(BasicBlock(input_channels, task_channels, downsample=downsample),
                                                BasicBlock(task_channels, task_channels))
            self.refinement = nn.ModuleDict(refinement)

        self.decoders = nn.ModuleDict({task: nn.Conv2d(task_channels, opt.TASKS.NUM_OUTPUT[task], 1) for task in self.auxilary_tasks})


    def forward(self, features_curr_scale, features_prev_scale=None):
        if features_prev_scale is not None: # Concat features that were propagated from previous scale
            x = {t: torch.cat((features_curr_scale, F.interpolate(features_prev_scale[t], scale_factor=2, mode='bilinear')), 1) for t in self.auxilary_tasks}

        else:
            x = {t: features_curr_scale for t in self.auxilary_tasks}

        # Refinement + Decoding
        out = {}
        for t in self.auxilary_tasks:
            out['features_%s' %(t)] = self.refinement[t](x[t])
            out[t] = self.decoders[t](out['features_%s' %(t)])

        return out


class FPM(nn.Module):
    """ Feature Propagation Module """
    def __init__(self, auxilary_tasks, per_task_channels):
        super(FPM, self).__init__()
        # General
        self.auxilary_tasks = auxilary_tasks
        self.N = len(self.auxilary_tasks)
        self.per_task_channels = per_task_channels
        self.shared_channels = int(self.N*per_task_channels)
        
        # Non-linear function f
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.shared_channels//4, 1, bias=False),
                                    nn.BatchNorm2d(self.shared_channels//4))
        self.non_linear = nn.Sequential(BasicBlock(self.shared_channels, self.shared_channels//4, downsample=downsample),
                                     BasicBlock(self.shared_channels//4, self.shared_channels//4),
                                     nn.Conv2d(self.shared_channels//4, self.shared_channels, 1))

        # Dimensionality reduction 
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.per_task_channels, 1, bias=False),
                                    nn.BatchNorm2d(self.per_task_channels))
        self.dimensionality_reduction = BasicBlock(self.shared_channels, self.per_task_channels,
                                                    downsample=downsample)

        # SEBlock
        self.se = nn.ModuleDict({task: SEBlock(self.per_task_channels) for task in self.auxilary_tasks})

    def forward(self, x):
        # Get shared representation
        concat = torch.cat([x['features_%s' %(task)] for task in self.auxilary_tasks], 1)
        B, C, H, W = concat.size()
        shared = self.non_linear(concat)
        mask = F.softmax(shared.view(B, C//self.N, self.N, H, W), dim = 2) # Per task attention mask
        shared = torch.mul(mask, concat.view(B, C//self.N, self.N, H, W)).view(B,-1, H, W)
        
        # Perform dimensionality reduction 
        shared = self.dimensionality_reduction(shared)

        # Per task squeeze-and-excitation
        out = {}
        for task in self.auxilary_tasks:
            out[task] = self.se[task](shared) + x['features_%s' %(task)]
        
        return out


class SpatialAttentionModule(nn.Module):
    def __init__(self, tasks, channels, feature_size, gamma):
        super(SpatialAttentionModule, self).__init__()
        self.tasks = tasks

        # Value projection for each task
        self.projection = nn.ModuleDict({task: nn.Conv2d(channels, channels, kernel_size=3, padding=1) for task in self.tasks})

        # Cross-Task Affinity Learning Modules
        self.task_fusion = nn.Conv2d(in_channels=len(self.tasks)*feature_size, out_channels=feature_size, kernel_size=3, stride=1, padding=1, groups=feature_size)

        # Blending parameter
        self.gamma = gamma

    def forward(self, x, return_m=False):
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
        M = self.task_fusion(M)

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

class MEMANet(nn.Module):
    """ 
        MTI-Net implementation based on HRNet backbone 
        https://arxiv.org/pdf/2001.06902.pdf
    """
    def __init__(self, opt, backbone, backbone_channels, backbone_dims, heads):
        super(MEMANet, self).__init__()
        # General
        self.tasks = opt.TASKS.NAMES
        self.auxilary_tasks = opt.TASKS.NAMES
        self.num_scales = len(backbone_channels)
        self.channels = backbone_channels      
        self.embedding_size = torch.tensor(backbone_dims, dtype=torch.int32).prod().item()
        self.gamma = opt['model_kwargs'].gamma
        self.out_channels = opt.TASKS.NUM_OUTPUT
        intermediate_channels = 128

        # Backbone
        self.backbone = backbone
        
        # Feature Propagation Module
        self.fpm_scale_3 = FPM(self.auxilary_tasks, self.channels[3])
        self.fpm_scale_2 = FPM(self.auxilary_tasks, self.channels[2])
        self.fpm_scale_1 = FPM(self.auxilary_tasks, self.channels[1])

        # Initial task predictions at multiple scales
        self.scale_0 = InitialTaskPredictionModule(opt, self.auxilary_tasks, self.channels[0] + self.channels[1], self.channels[0])
        self.scale_1 = InitialTaskPredictionModule(opt, self.auxilary_tasks, self.channels[1] + self.channels[2], self.channels[1])
        self.scale_2 = InitialTaskPredictionModule(opt, self.auxilary_tasks, self.channels[2] + self.channels[3], self.channels[2])
        self.scale_3 = InitialTaskPredictionModule(opt, self.auxilary_tasks, self.channels[3], self.channels[3])

        # Initial predisction fusion
        self.scale_fusion = nn.ModuleDict({task: HighResolutionFuse(self.channels, 256) for task in self.tasks})

        # Distillation at multiple scales
        self.ctal = AffinityLearningModule(self.auxilary_tasks, sum(backbone_channels), intermediate_channels, self.embedding_size, self.gamma)
        
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
        
        # Predictions at multiple scales
            # Scale 3
        x_3 = self.scale_3(x[3])
        x_3_fpm = self.fpm_scale_3(x_3)
            # Scale 2
        x_2 = self.scale_2(x[2], x_3_fpm)
        x_2_fpm = self.fpm_scale_2(x_2)
            # Scale 1
        x_1 = self.scale_1(x[1], x_2_fpm)
        x_1_fpm = self.fpm_scale_1(x_1)
            # Scale 0
        x_0 = self.scale_0(x[0], x_1_fpm)
        
        out['deep_supervision'] = {'scale_0': x_0, 'scale_1': x_1, 'scale_2': x_2, 'scale_3': x_3}  

        x_f = {}
        for task in self.tasks:
            x_f[f'features_{task}'] = self.scale_fusion[task]([x_0[f'features_{task}'],x_1[f'features_{task}'],x_2[f'features_{task}'],x_3[f'features_{task}']])
        x_out = self.ctal(x_f)

        # Feature aggregation
        for t in self.tasks:
            out[t] = F.interpolate(self.heads[t](x_out[f'features_{t}']), img_size, mode='bilinear')
            
        return out

def concat(x, y, tasks):
    for t in tasks:
        x[f'features_{t}'] = torch.cat((x[f'features_{t}'], y[f'features_{t}']), dim=1)
    return x