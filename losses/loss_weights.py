import torch
import torch.nn as nn

class Scalarization(nn.Module):
    def __init__(self, opt):
        super(Scalarization, self).__init__()
        self.tasks = opt.TASKS.NAMES
        self.weights = opt['loss_kwargs']['loss_weights']
    
    def forward(self, losses):
        intial_loss = sum([self.weights[task]*losses['initial'][task] for task in self.tasks])
        final_loss = sum([self.weights[task]*losses['final'][task] for task in self.tasks])
        total_loss = intial_loss + final_loss

        return total_loss


class Uncertainty(nn.Module):
    def __init__(self, opt):
        super(Uncertainty, self).__init__()
        self.tasks = opt.TASKS.NAMES
        self.weights = nn.ParameterDict({t: nn.Parameter(torch.tensor(-0.5, dtype=torch.float)) for t in self.tasks})
    
    def forward(self, losses):
        initial_loss = 0
        final_loss = 0
        for task in self.tasks:
            initial_loss += losses['initial'][task]/(2*torch.exp(self.weights[task])) + 0.5*self.weights[task]
            final_loss += losses['final'][task]/(2*torch.exp(self.weights[task])) + 0.5*self.weights[task]

        return initial_loss + final_loss