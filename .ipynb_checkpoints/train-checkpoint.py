import argparse
from tqdm import tqdm
import visdom
import os

from multiprocessing import cpu_count

import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.optim as optim

from models.generator import Generator
from models.discriminator import Discriminator

from criterion import D_Criterion_adv, Criterion_h_rec, R1_Penalty
from criterion import G_Criterion_adv
from utils import Visdom
from utils import FID

from torch_ema import ExponentialMovingAverage
from copy import deepcopy

from dataset import TherDataset

def parse():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # training setting
    parser.add_argument('--model_name', default='test')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--warup_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--g_lr', type=float, default=5e-5)
    parser.add_argument('--d_lr', type=float, default=2e-4)
    # training strategy
    parser.add_argument('--d_updates', type=int, default=1)
    parser.add_argument('--gp_method', default='R1')
    parser.add_argument('--d_rec_weight', type=int, default=2)
    parser.add_argument('--d_gp_weight', type=int, default=5)
    parser.add_argument('--g_rec_weight', type=int, default=2)
    # dataset setting
    parser.add_argument('--h_size_h', type=int, default=5)
    parser.add_argument('--h_size_w', type=int, default=8)
    parser.add_argument('--rgb_size_h', type=int, default=40)
    parser.add_argument('--rgb_size_w', type=int, default=64)
    # I/O setting
    parser.add_argument('--rgb_npy', default="../lrt-human/rgb.npy")
    parser.add_argument('--thermal_npy', default="../lrt-human/thermal.npy")
    parser.add_argument('--saving_folder', default='ckpts')
    # 
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpus', type=int, default=4)
    # model setting
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--g_ther_conv_dim', type=int, default=64)
    parser.add_argument('--g_conv_dim', type=int, default=128)
    parser.add_argument('--d_ther_conv_dim', type=int, default=128)
    parser.add_argument('--d_conv_dim', type=int, default=128)
    # Visdom setting
    parser.add_argument('--port', type=int, default=1203)
    parser.add_argument('--env', default="test")
    parser.add_argument('--visual_loss_step', type=int, default=10)
    parser.add_argument('--visual_output_step', type=int, default=50)
    # Mode
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--resume', action="store_true")
    
    return parser.parse_args()

class Model_factory():
    def __init__(self, args, vis):
        self.args = args
        self.vis = vis
        self.device = args.device
        
        trainDataset = TherDataset(rgb_npy_path=args.rgb_npy, thermal_npy_path=args.thermal_npy)
        self.trainDataloader = DataLoader(dataset=trainDataset,
                                          batch_size=args.batch_size,
                                          shuffle=True, 
                                          num_workers=cpu_count())
        
        self.G = Generator(h_size_h=args.h_size_h, h_size_w=args.h_size_w,
                           z_dim=args.z_dim, ther_conv_dim=args.g_ther_conv_dim, conv_dim=args.g_conv_dim)
        self.D = Discriminator(h_size_h=args.h_size_h, h_size_w=args.h_size_w,
                               ther_conv_dim=args.d_ther_conv_dim, conv_dim=args.d_conv_dim)
        
        self.G = DataParallel(self.G, device_ids=[i for i in range(self.args.gpus)]).to(self.device)
        self.D = DataParallel(self.D, device_ids=[i for i in range(self.args.gpus)]).to(self.device)
        
        self.G_ema = ExponentialMovingAverage(self.G.parameters(), decay=0.995)
        self.D_ema = ExponentialMovingAverage(self.D.parameters(), decay=0.995)

        self.G_optimizer = optim.AdamW(self.G.parameters(), lr=args.g_lr)
        self.D_optimizer = optim.AdamW(self.D.parameters(), lr=args.d_lr)
        
        self.D_adv_criterion = D_Criterion_adv(mode='hinge')
        self.D_h_rec_criterion = Criterion_h_rec(mode='hinge', hinge_threshold=0.05)
        if args.gp_method == 'R1':
            self.D_gp_criterion = R1_Penalty()
            
        self.G_adv_criterion = G_Criterion_adv()
        self.G_h_rec_criterion = Criterion_h_rec(mode='hinge', hinge_threshold=0.05)
        
        self.FID = FID(feature=2048)
        self.saving_folder = os.path.join(args.saving_folder, args.model_name)
        
    def train_D(self):
        b, c, h, w = self.rgb.shape
        z = torch.randn([b, self.args.z_dim]).to(self.device)
        fake_imgs = self.G(self.thermal, z)

        fake_labels, fake_rec_thermal = self.D(fake_imgs)
        real_labels, real_rec_thermal = self.D(self.rgb)
        
        loss_d_adv = self.D_adv_criterion(real_labels, fake_labels)
        loss_d_rec_real = self.D_h_rec_criterion(self.thermal, real_rec_thermal)
        loss_d_rec_fake = self.D_h_rec_criterion(self.thermal, fake_rec_thermal)
        loss_d_gp = self.D_gp_criterion(self.rgb, self.D)

        d_loss = loss_d_adv + self.args.d_rec_weight*(loss_d_rec_fake+loss_d_rec_real) + self.args.d_gp_weight*loss_d_gp
        
        self.D_optimizer.zero_grad()
        d_loss.backward()
        self.D_optimizer.step()
        self.D_ema.update()
        
        loss = {"D_loss": d_loss.item(), "D_adv": loss_d_adv.item(), "D_rec_real": loss_d_rec_real.item(), "D_rec_fake": loss_d_rec_fake.item(), "D_gp": loss_d_gp.item()}
        
        return loss
    
    def train_G(self):
        b, c, h, w = self.rgb.shape
        z = torch.randn([b, self.args.z_dim]).to(self.device)
        fake_imgs = self.G(self.thermal, z)
        self.fake_rgb = fake_imgs
        
        fake_labels, fake_rec_thermal = self.D(fake_imgs)
        self.fake_rec_thermal = fake_rec_thermal
        
        loss_g_adv = self.G_adv_criterion(fake_labels)
        loss_g_rec = self.G_h_rec_criterion(self.thermal, fake_rec_thermal)
        
        g_loss = loss_g_adv + self.args.g_rec_weight*loss_g_rec
        
        self.G_optimizer.zero_grad()
        g_loss.backward()
        self.G_optimizer.step()
        self.G_ema.update()
        
        loss = {"G_loss": g_loss.item(), "G_adv": loss_g_adv.item(), "G_rec": loss_g_rec.item()}
        
        return loss
    
    def draw_visualization(self, fake_rec_thermal, thermal, fake_rgb, rgb, cols=4):
        fake_thermal = fake_rec_thermal[:cols]
        real_thermal = thermal[:cols]
        fake_rgb = fake_rgb[:cols]
        real_rgb = rgb[:cols]
        
        return torch.cat([fake_thermal, real_thermal], dim=0), torch.cat([fake_rgb, real_rgb], dim=0)
        
    def train(self, e):
        self.G.train()
        self.D.train()
        l = len(self.trainDataloader)
        pbar = tqdm(self.trainDataloader)
        for i, (rgb, thermal) in enumerate(pbar):
            self.rgb = rgb.to(self.device)
            self.thermal = thermal.to(self.device)
            
            for _ in range(self.args.d_updates):
                D_loss = self.train_D()
                
            G_loss = self.train_G()
                
            pbar.set_description("D_loss: {:.05f}, G_loss: {:.05f}".format(D_loss['D_loss'], G_loss['G_loss']))
                
            if i % self.args.visual_loss_step == 0:
                self.vis.Line(loss_type="G_loss", win='Loss', loss=G_loss['G_loss'], step=l*(e-1)+i)
                self.vis.Line(loss_type="D_loss", win='Loss', loss=D_loss['D_loss'], step=l*(e-1)+i)
                
            if i % self.args.visual_output_step == 0:
                vis_thermal, vis_rgb = self.draw_visualization(fake_rec_thermal=self.fake_rec_thermal, thermal=self.thermal,
                                                               fake_rgb=self.fake_rgb, rgb=self.rgb)
                self.vis.Images(images=vis_thermal, win='Current Thermal', ncol=4, unormalize=True)
                self.vis.Images(images=vis_rgb, win='Current RGB', ncol=4, unormalize=True)
        
    def validate(self, e):
        rgb, thermal = next(iter(self.trainDataloader))
        rgb = rgb.to(self.device)
        thermal = thermal.to(self.device)
        b, c, h, w = rgb.shape
        z = torch.randn([b, self.args.z_dim]).to(self.device)
        with self.G_ema.average_parameters():
            fake_imgs = self.G(thermal, z)
        with self.D_ema.average_parameters():
            fake_labels, fake_rec_thermal = self.D(fake_imgs)
        vis_thermal, vis_rgb = self.draw_visualization(fake_rec_thermal=fake_rec_thermal, thermal=thermal,
                                                       fake_rgb=fake_imgs, rgb=rgb)
        self.vis.Images(images=vis_thermal, win="Epoch-{}-thermal".format(e), ncol=4, unormalize=True)
        self.vis.Images(images=vis_rgb, win="Epoch-{}-rgb".format(e), ncol=4, unormalize=True)
        
    def calc_fid(self):
        pbar = tqdm(self.trainDataloader)
        for i, (rgb, thermal) in enumerate(pbar):
            rgb = rgb.to(self.device)
            thermal = thermal.to(self.device)
            b, c, h, w = rgb.shape
            z = torch.randn([b, self.args.z_dim]).to(self.device)
            with self.G_ema.average_parameters():
                fake_imgs = self.G(thermal, z)
            with self.D_ema.average_parameters():
                fake_labels, fake_rec_thermal = self.D(fake_imgs)

            self.FID.update(real=rgb, fake=fake_imgs)
        fid = self.FID.compute()
        print("FID:", fid)
            
    def save(self, e):
        print("Saving Model...")
        if not os.path.isdir(self.saving_folder): os.makedirs(self.saving_folder)
        ckpt = {
            "G_parameter": self.G.state_dict(),
            "D_parameter": self.D.state_dict(),
            "epoch": e,
            "G_optimizer": self.G_optimizer.state_dict(),
            "D_optimizer": self.D_optimizer.state_dict(),
            "G_ema": self.G_ema.state_dict(),
            "D_ema": self.D_ema.state_dict()
        }
        torch.save(ckpt, '{}/checkpoint.ckpt'.format(self.saving_folder))
        
        G_ema_model = deepcopy(self.G)
        D_ema_model = deepcopy(self.D)
        self.G_ema.copy_to(G_ema_model.parameters())
        self.D_ema.copy_to(D_ema_model.parameters())
        
        torch.save(G_ema_model, '{}/G.pt'.format(self.saving_folder))
        torch.save(D_ema_model, '{}/D.pt'.format(self.saving_folder))

    def load(self):
        print("Loading Model...")
        ckpt = torch.load('{}/checkpoint.ckpt'.format(self.saving_folder))
        self.G.load_state_dict(ckpt['G_parameter'])
        self.D.load_state_dict(ckpt['D_parameter'])
        self.G_optimizer.load_state_dict(ckpt['G_optimizer'])
        self.D_optimizer.load_state_dict(ckpt['D_optimizer'])
        self.G_ema.load_state_dict(ckpt['G_ema'])
        self.D_ema.load_state_dict(ckpt['D_ema'])
        
        epoch = ckpt['epoch']
        return epoch
        
def main():
    args = parse()
    vis = Visdom(args.env, args.port)
    Model = Model_factory(args, vis)
    
    start_epoch = Model.load()+1 if args.resume else 1
    if args.train:
        for e in range(start_epoch, args.epochs+1):
            print("Epoch: {}".format(e))
            Model.train(e)
            
            if e % 5 == 0:
                print("Validating...")
                Model.validate(e)
                Model.save(e)

    if args.test:
        Model.load()
        Model.validate(e='Test')
        Model.calc_fid()
        
    
if __name__ == '__main__':
    main()