import torch
from torch import nn

from einops import rearrange

from .modules import ResBlock_up, Spade_BN, Spade_sr_up

class Generator(nn.Module):
    def __init__(self, h_size_h=5, h_size_w=8, z_dim=128, ther_conv_dim=128, conv_dim=128):
        super().__init__()
        self.h_size_h = h_size_h
        self.h_size_w = h_size_w
        
        self.ther_res_up1 = ResBlock_up(in_ch=1, out_ch=ther_conv_dim, scale=1)
        self.ther_res_up2 = ResBlock_up(in_ch=ther_conv_dim, out_ch=ther_conv_dim, scale=2)
        self.ther_res_up3 = ResBlock_up(in_ch=ther_conv_dim, out_ch=ther_conv_dim, scale=2)
        self.ther_res_up4 = ResBlock_up(in_ch=ther_conv_dim, out_ch=ther_conv_dim, scale=2)
        
        self.f = nn.Linear(z_dim, h_size_h*h_size_w*conv_dim*4)
        self.spade_sr_up1 = Spade_sr_up(h_ch=ther_conv_dim, in_ch=conv_dim*4, out_ch=conv_dim*4, scale=2)
        self.spade_sr_up2 = Spade_sr_up(h_ch=ther_conv_dim, in_ch=conv_dim*4, out_ch=conv_dim*2, scale=2)
        self.spade_sr_up3 = Spade_sr_up(h_ch=ther_conv_dim, in_ch=conv_dim*2, out_ch=conv_dim, scale=2)
        
        self.spade_bn = Spade_BN(h_ch=ther_conv_dim, in_ch=conv_dim, out_ch=conv_dim)
        self.relu = nn.ReLU()
        
        self.out_conv = nn.Conv2d(conv_dim, 3, kernel_size=3, padding=1)
        
    def forward(self, thermal, noise):
        h_c1 = self.ther_res_up1(thermal)
        h_c2 = self.ther_res_up2(h_c1)
        h_c3 = self.ther_res_up2(h_c2)
        h_c4 = self.ther_res_up2(h_c3)
        
        x = self.f(noise)
        x = rearrange(x, "b (h w d) -> b d h w", h=self.h_size_h, w=self.h_size_w, d=128*4)
        x = self.spade_sr_up1(x, h_c1, h_c2)
        x = self.spade_sr_up2(x, h_c2, h_c3)
        x = self.spade_sr_up3(x, h_c3, h_c4)
        
        x = self.spade_bn(x, h_c4)
        x = self.relu(x)
        
        out = self.out_conv(x)
        return out
        
        
        
if __name__ == '__main__':
    G = Generator()
    G(torch.ones([1, 1, 5, 8]), torch.ones([1, 128]))