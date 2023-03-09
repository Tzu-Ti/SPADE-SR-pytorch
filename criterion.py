import torch
from torch import nn, autograd

class D_Criterion_adv(nn.Module):
    def __init__(self, mode='hinge'):
        super().__init__()
        self.mode = mode
        self.relu = nn.ReLU()
        
    def forward(self, real_labels, fake_labels):
        if self.mode == 'hinge':
            return torch.mean(self.relu(1 - real_labels)) + torch.mean(self.relu(1 + fake_labels))
        else:
            raise
        
class Criterion_h_rec(nn.Module):
    def __init__(self, mode='hinge', hinge_threshold=0.05):
        super().__init__()
        self.mode = mode
        self.hinge_threshold = hinge_threshold
        self.relu = nn.ReLU()
        
    def forward(self, thermal_GT, thermal_rec):
        if self.mode == 'hinge':
            return torch.mean(self.relu(torch.abs(thermal_GT - thermal_rec) - self.hinge_threshold))
        
class R1_Penalty(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, rgb, D):
        b, c, h, w = rgb.shape
        rgb.requires_grad = True
        label, ther_rec = D(rgb)
        grad = autograd.grad(outputs=label, inputs=rgb,
                             grad_outputs=torch.ones(label.shape).cuda(),
                             create_graph=True, retain_graph=True)[0]
        grad = grad.view(b, -1).norm(2, dim=1) ** 2
        return grad.mean()
    
class G_Criterion_adv(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, fake_labels):
        return -torch.mean(fake_labels)