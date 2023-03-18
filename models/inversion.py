import torch
from torch import nn

from .modules import ResBlock_down

class Inversion(nn.Module):
    def __init__(self, img_size_h=40, img_size_w=64, e_conv_dim=64, z_dim=128):
        super().__init__()
        
        self.res_down1 = ResBlock_down(in_ch=3, out_ch=e_conv_dim, scale=1, use_bn=True)
        self.res_down2 = ResBlock_down(in_ch=e_conv_dim, out_ch=e_conv_dim*2, scale=2, use_bn=True)
        self.res_down3 = ResBlock_down(in_ch=e_conv_dim*2, out_ch=e_conv_dim*4, scale=2, use_bn=True)
        self.res_down4 = ResBlock_down(in_ch=e_conv_dim*4, out_ch=e_conv_dim*8, scale=2, use_bn=True)
        
        self.BN = nn.BatchNorm2d(e_conv_dim*8)
        self.relu = nn.ReLU()
        
        self.flatten = nn.Flatten()
        self.f = nn.Linear((e_conv_dim * 8) * img_size_h * img_size_w // (2**3) // (2**3), z_dim, bias=False)
        self.BN_z = nn.BatchNorm1d(z_dim)
        
    def forward(self, x):
        x = self.res_down1(x)
        x = self.res_down2(x)
        x = self.res_down3(x)
        x = self.res_down4(x)
        
#         x = self.BN(x)
        x = self.relu(x)
        
        z = self.flatten(x)
        z = self.f(z)
#         z = self.BN_z(z)
        
        return z
    
if __name__ == '__main__':
    INV = Inversion()
    z = INV(torch.ones([2, 3, 40, 64]))
    print(z.shape)
        