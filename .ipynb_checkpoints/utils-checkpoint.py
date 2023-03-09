import visdom
import torch

class Visdom():
    def __init__(self, env, port):
        self.env = env
        self.port = port
        self.vis = visdom.Visdom(env=env, port=port)

    def Line(self, loss, step, win='Loss', loss_type='G_loss'):
        self.vis.line(win=win, Y=[loss], X=[step], env=self.env, update='append', name=loss_type, opts={'title': win})
    
    def Images(self, images, win, ncol=4, unormalize=True):
        if unormalize:
            images = images.clip(-1, 1)
            images = ((images + 1) / 2 * 255.0).to(torch.uint8)
        
        self.vis.images(images, win=win, env=self.env, nrow=ncol, opts={'title': win, 'width': 600, 'height': 250}) #nrow means number of images in a row