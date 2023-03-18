from torch import nn

class ResBlock_up(nn.Module):
    def __init__(self, in_ch=1, out_ch=128, scale=2, use_bn=False):
        super().__init__()
        self.use_bn = use_bn
        self.scale = scale
        
        self.skip_upsample = nn.Upsample(scale_factor=scale)
        self.skip_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        
        self.BN1 = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU()
        self.up = nn.Upsample(scale_factor=scale)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.BN2 = nn.BatchNorm2d(in_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
       
    def forward(self, x):
        skip = self.skip_upsample(x)
        skip = self.skip_conv(skip)
        
        x = self.BN1 if self.use_bn else x
        x = self.relu(x)
        x = self.up(x) if self.scale > 1 else x
        x = self.conv1(x)
        x = self.BN2 if self.use_bn else x
        x = self.relu(x)
        x = self.conv2(x)
        
        return x + skip
    
class ResBlock_down(nn.Module):
    def __init__(self, in_ch=1, out_ch=128, scale=2, use_bn=False):
        super().__init__()
        self.in_ch = in_ch
        self.use_bn = use_bn
        self.scale = scale
        
        self.skip_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.skip_avgpool = nn.AvgPool2d(scale)
        
        self.BN1 = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.BN2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.avgpool = nn.AvgPool2d(scale)
        
    def forward(self, x):
        if self.in_ch == 3:
            skip = self.skip_avgpool(x) if self.scale > 1 else x
            skip = self.skip_conv(skip)
        else:
            skip = self.skip_conv(x)
            skip = self.skip_avgpool(skip) if self.scale > 1 else skip
            
        x = self.BN1(x) if self.use_bn else x
        x = self.relu(x) if self.in_ch == 3 else x
        x = self.conv1(x)
        
        x = self.BN2(x) if self.use_bn else x
        x = self.relu(x)
        x = self.conv2(x)
        
        x = self.avgpool(x) if self.scale > 1 else x
        
        return x + skip
        
class Spade_BN(nn.Module):
    def __init__(self, h_ch, in_ch, out_ch):
        super().__init__()
        
        self.BN = nn.BatchNorm2d(in_ch, affine=False)
        self.gamma_conv = nn.Conv2d(h_ch, out_ch, kernel_size=3, padding=1)
        self.beta_conv = nn.Conv2d(h_ch, out_ch, kernel_size=3, padding=1)
        
    def forward(self, x, h_feat):
        x = self.BN(x)
        gamma = self.gamma_conv(h_feat)
        beta = self.beta_conv(h_feat)
        
        out = x * (1 + gamma) + beta
        return out
        
class Spade_sr_up(nn.Module):
    def __init__(self, h_ch, in_ch, out_ch, scale=2):
        super().__init__()
        
        self.skip_upsample = nn.Upsample(scale_factor=scale)
        self.skip_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        
        self.act = nn.ReLU()
        self.spade_bn1 = Spade_BN(h_ch=h_ch, in_ch=in_ch, out_ch=in_ch)
        self.upsample = nn.Upsample(scale_factor=scale)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        
        self.spade_bn2 = Spade_BN(h_ch=h_ch, in_ch=out_ch, out_ch=out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        
    def forward(self, x, h_c1, h_c2):
        skip = self.skip_upsample(x)
        skip = self.skip_conv(skip)
        
        x = self.spade_bn1(x, h_c1)
        x = self.act(x)
        x = self.upsample(x)
        x = self.conv1(x)
        
        x = self.spade_bn2(x, h_c2)
        x = self.conv2(x)
        
        return x + skip
        
        