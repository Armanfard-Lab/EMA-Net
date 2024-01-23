#
# Authors: Dimitrios Sinodinos
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

"""
    Implementation of PAP-Net.
    https://arxiv.org/abs/1906.03525
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import Bottleneck
from models.layers import SEBlock, SABlock

def compute_affinity_matrix(features):
    """
    Compute the affinity matrix for the given features.
    
    Args:
    - features (torch.Tensor): feature maps of shape (B, C, H, W) 
      
    Returns:
    - affinity_matrix (torch.Tensor): affinity matrix of shape (B, H*W, H*W)
    """
    
    B, C, H, W = features.size()
    
    # Reshape features to (B, H*W, C)
    features = features.view(B, C, -1).transpose(1,2)
    
    # Normalize the features
    features = F.normalize(features, dim=1)
    
    # Compute the dot product to get the affinity matrix
    affinity_matrix = torch.bmm(features, features.transpose(1, 2))

    # Row normalization
    affinity_matrix /= affinity_matrix.sum(1, keepdim=True)
    
    return affinity_matrix

def diffuse(features, joint_affinity, beta=0.05, num_iterations=3):
    """
    Apply the diffusion module to propagate patterns using the joint affinity matrix iteratively.

    Args:
    - features (torch.Tensor): Feature maps of shape (B, C, H, W)
    - joint_affinity (torch.Tensor): Joint affinity matrix of shape (B, H*W, H*W)
    - beta (float): Diffusion parameter, it determines the strength of propagation
    - num_iterations (int): Number of times to apply the diffusion process iteratively

    Returns:
    - diffused_features (torch.Tensor): Diffused feature maps of shape (B, C, H, W)
    """
    
    B, C, H, W = features.size()
    
    # Reshape features to (B, H*W, C)
    features_reshaped = features.view(B, C, -1).transpose(1,2)
    
    diffused_features = features_reshaped.clone()

    # Iteratively apply the diffusion process
    for _ in range(num_iterations):
        diffused_features = beta * torch.bmm(joint_affinity, diffused_features) + (1 - beta) * features_reshaped
    
    # Reshape diffused features back to (B, C, H, W)
    diffused_features = diffused_features.transpose(1,2).view(B, C, H, W)
    
    return diffused_features


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


class AffinityLearningModule(nn.Module):
    def __init__(self, auxilary_tasks, channels):
        super(AffinityLearningModule, self).__init__()
        self.tasks = auxilary_tasks

        # Conv layer to reduce channels from 2C to C
        self.conv = nn.ModuleDict({task: nn.Conv2d(channels, channels//2, 1) for task in self.tasks})

    def forward(self, x, mat_size):
        out = {}
        for task in self.tasks:
            features = self.conv[task](x[f'features_{task}'])
            features = F.interpolate(features, mat_size, mode='bilinear')
            out[f'features_{task}'] = features
            out[f'matrix_{task}'] = compute_affinity_matrix(features)
        return out


class MultiTaskDiffusionModule(nn.Module):
    def __init__(self, tasks, auxilary_tasks, beta, num_iters):
        super(MultiTaskDiffusionModule, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.beta = beta
        self.num_iters = num_iters
        self.weights = nn.ParameterDict({t: nn.Parameter(torch.ones(len(tasks))/len(tasks)) for t in self.tasks})

    # TODO: Impliment subsampling
    def forward(self, x):
        out = {}
        
        for task in self.tasks:
            # compute combined matrices
            task_weights = F.softmax(self.weights[task], dim=0) # make sure weights sum up to 1
            combined_affinity_matrix = sum([w * x[f'matrix_{t}'] for t, w in zip(self.tasks, task_weights)])

            # diffuse
            features = x[f'features_{task}']
            out[task] = diffuse(features, combined_affinity_matrix, self.beta, self.num_iters)
        
        return out        


class PAPNet(nn.Module):
    def __init__(self, opt, backbone, backbone_channels):
        super(PAPNet, self).__init__()
        # General
        self.tasks = opt.TASKS.NAMES
        self.auxilary_tasks = opt.TASKS.NAMES
        self.channels = backbone_channels
        self.beta = opt['model_kwargs']['diffusion'].beta
        self.num_iters = opt['model_kwargs']['diffusion'].num_iters
        self.out_channels = opt.TASKS.NUM_OUTPUT

        # Backbone
        self.backbone = backbone

        # Task-specific heads for initial prediction 
        self.initial_task_prediction_heads = InitialTaskPredictionModule(self.out_channels, self.auxilary_tasks, self.channels)

        # Cross-task proagation
        self.affinity_learning_module = AffinityLearningModule(self.auxilary_tasks, 256)

        # Task-specific propagation
        self.diffusion_module = MultiTaskDiffusionModule(self.tasks, self.auxilary_tasks, self.beta, self.num_iters)

        # Task-specific heads for final prediction
        heads = {}
        for task in self.tasks:
            bottleneck1 = Bottleneck(128, 128//4, downsample=None)
            bottleneck2 = Bottleneck(128, 128//4, downsample=None)
            conv_out_ = nn.Conv2d(128, self.out_channels[task], 1)
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
        x = self.affinity_learning_module(x, mat_size=(img_size[0]//4,img_size[1]//4))

        # Propagate patterns using join affinities
        x = self.diffusion_module(x)

        # Make final prediction with task-specific heads
        for task in self.tasks:
            out[task] = F.interpolate(self.heads[task](x[task]), img_size, mode='bilinear')

        return out