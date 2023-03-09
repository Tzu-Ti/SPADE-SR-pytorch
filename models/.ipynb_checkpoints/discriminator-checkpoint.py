import torch
from torch import nn

from .modules import ResBlock_down

class Discriminator(nn.Module):
    def __init__(self, h_size_h=5, h_size_w=8, ther_conv_dim=128, conv_dim=128):
        super().__init__()
        
        self.res_down1 = ResBlock_down(in_ch=3, out_ch=conv_dim, scale=2)
        self.res_down2 = ResBlock_down(in_ch=conv_dim, out_ch=conv_dim*2, scale=2)
        self.res_down3 = ResBlock_down(in_ch=conv_dim*2, out_ch=conv_dim*4, scale=2)
        
        self.relu = nn.ReLU()
        
        self.flatten = nn.Flatten()
        self.f = nn.Linear(conv_dim*4*h_size_h*h_size_w, 1)
        
        self.ther_res_down = ResBlock_down(in_ch=conv_dim*4, out_ch=ther_conv_dim, scale=1)
        self.ther_conv = nn.Conv2d(ther_conv_dim, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        c1 = self.res_down1(x)
        c2 = self.res_down2(c1)
        c3 = self.res_down3(c2)
        
        latent = self.relu(c3)
        latent = self.flatten(latent)
        label = self.f(latent)
        
        ther_rec = self.ther_res_down(c3)
        ther_rec = self.relu(ther_rec)
        ther_rec = self.ther_conv(ther_rec)

        return label, ther_rec
        
        
if __name__ == '__main__':
    D = Discriminator()
    D(torch.ones([1, 3, 40, 64]))
        