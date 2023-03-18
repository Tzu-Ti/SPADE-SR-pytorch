import visdom
import torch

def Unormalize(x):
    x = x.clip(-1, 1)
    x = ((x + 1) / 2 * 255.0).to(torch.uint8)
    return x

class Visdom():
    def __init__(self, env, port):
        self.env = env
        self.port = port
        self.vis = visdom.Visdom(env=env, port=port)

    def Line(self, loss, step, win='Loss', loss_type='G_loss'):
        self.vis.line(win=win, Y=[loss], X=[step], env=self.env, update='append', name=loss_type, opts={'title': win})
    
    def Images(self, images, win, ncol=4, unormalize=True):
        images = Unormalize(images) if unormalize else images
        
        self.vis.images(images, win=win, env=self.env, nrow=ncol, opts={'title': win, 'width': 600, 'height': 250}) #nrow means number of images in a row
        
        
from torchmetrics.image.fid import FrechetInceptionDistance
from torch import nn
class FID():
    def __init__(self, feature=2048, device='cuda'):
        self.fid = FrechetInceptionDistance(feature=feature).to(device)
        
    def update(self, real, fake):
        real = Unormalize(real)
        fake = Unormalize(fake)

        self.fid.update(real, real=True)
        self.fid.update(fake, real=False)
        
    def compute(self):
        return self.fid.compute()
        
        
    