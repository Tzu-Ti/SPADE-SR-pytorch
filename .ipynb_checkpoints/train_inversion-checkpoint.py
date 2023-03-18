import argparse
from tqdm import tqdm
import visdom
import os

from multiprocessing import cpu_count

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.optim as optim

from models.inversion import Inversion

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
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--warup_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--inv_lr', type=float, default=5e-5)
    parser.add_argument('--d_lr', type=float, default=1e-5)
    # training strategy
    parser.add_argument('--d_updates', type=int, default=1)
    parser.add_argument('--gp_method', default='R1')
    parser.add_argument('--d_rec_weight', type=int, default=0)
    parser.add_argument('--d_gp_weight', type=int, default=0)
    parser.add_argument('--inv_h_rec_weight', type=int, default=0)
    parser.add_argument('--inv_adv_weight', type=float, default=0.1)
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
    parser.add_argument('--d_ther_conv_dim', type=int, default=128)
    parser.add_argument('--d_conv_dim', type=int, default=128)
    parser.add_argument('--e_conv_dim', type=int, default=64)
    # Visdom setting
    parser.add_argument('--port', type=int, default=1203)
    parser.add_argument('--env', default="test")
    parser.add_argument('--visual_loss_step', type=int, default=10)
    parser.add_argument('--visual_output_step', type=int, default=100)
    # Mode
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action="store_true")
    
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
        
        self.saving_folder = os.path.join(args.saving_folder, args.model_name)
        self.load()
        
        self.INV = Inversion(img_size_h=args.rgb_size_h, img_size_w=args.rgb_size_w, 
                             e_conv_dim=args.e_conv_dim, z_dim=args.z_dim)
        self.INV = DataParallel(self.INV, device_ids=[i for i in range(self.args.gpus)]).to(self.device)
        
        self.INV_ema = ExponentialMovingAverage(self.INV.parameters(), decay=0.995)
        self.D_ema = ExponentialMovingAverage(self.D.parameters(), decay=0.995)
        
        self.INV_optimizer = optim.AdamW(self.INV.parameters(), lr=args.inv_lr)
        self.D_optimizer = optim.AdamW(self.D.parameters(), lr=args.d_lr)
        
        self.inv_rgb_criterion = nn.L1Loss()
        self.inv_h_rec_criterion = Criterion_h_rec(mode='hinge', hinge_threshold=0)
        self.inv_adv_criterion = G_Criterion_adv()
        
        self.D_adv_criterion = D_Criterion_adv(mode='hinge')
        self.D_h_rec_criterion = Criterion_h_rec(mode='hinge', hinge_threshold=0.05)
        if args.gp_method == 'R1':
            self.D_gp_criterion = R1_Penalty()
        
        self.FID = FID(feature=2048)
        
    def draw_visualization(self, fake_rec_thermal, thermal, fake_rgb, rgb, cols=4):
        fake_thermal = fake_rec_thermal[:cols]
        real_thermal = thermal[:cols]
        fake_rgb = fake_rgb[:cols]
        real_rgb = rgb[:cols]
        
        return torch.cat([fake_thermal, real_thermal], dim=0), torch.cat([fake_rgb, real_rgb], dim=0)
        
    def train_INV(self):
        inv_z = self.INV(self.rgb)
        rec_images = self.G(self.thermal, inv_z)
        self.rec_images = rec_images
            
        fake_labels, fake_rec_thermal = self.D(rec_images)
        self.fake_rec_thermal = fake_rec_thermal
        
        loss_inv_rec = self.inv_rgb_criterion(rec_images, self.rgb)
        loss_inv_h_rec = self.inv_h_rec_criterion(fake_rec_thermal, self.thermal)
        loss_inv_adv = self.inv_adv_criterion(fake_labels)

        inv_loss = loss_inv_rec + self.args.inv_h_rec_weight*loss_inv_h_rec + self.args.inv_adv_weight*loss_inv_adv
        
        self.INV_optimizer.zero_grad()
        inv_loss.backward()
        self.INV_optimizer.step()
        self.INV_ema.update()
        
        loss = {"INV_loss": inv_loss.item()}
        
        return loss
    
    def train_D(self):
        inv_z = self.INV(self.rgb)

        rec_images = self.G(self.thermal, inv_z)
            
        fake_labels, fake_rec_thermal = self.D(rec_images)
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
        
    def train(self, e):
        self.INV.train()
        self.G.eval()
        self.D.train()
        l = len(self.trainDataloader)
        pbar = tqdm(self.trainDataloader)
        for i, (rgb, thermal) in enumerate(pbar):
            self.rgb = rgb.to(self.device)
            self.thermal = thermal.to(self.device)
            
            for _ in range(self.args.d_updates):
                INV_loss = self.train_INV()
            D_loss = self.train_D()
#             D_loss = {"D_loss": 0}
            
            
            pbar.set_description("D_loss: {:.05f}, INV_loss: {:.05f}".format(D_loss['D_loss'], INV_loss['INV_loss']))
                
            if i % self.args.visual_loss_step == 0:
                self.vis.Line(loss_type="INV_loss", win='INV-Loss', loss=INV_loss['INV_loss'], step=l*(e-1)+i)
                self.vis.Line(loss_type="D_loss", win='INV-Loss', loss=D_loss['D_loss'], step=l*(e-1)+i)
                
            if i % self.args.visual_output_step == 0:
                vis_thermal, vis_rgb = self.draw_visualization(fake_rec_thermal=self.fake_rec_thermal, thermal=self.thermal,
                                                               fake_rgb=self.rec_images, rgb=self.rgb)
                self.vis.Images(images=vis_thermal, win='Current Thermal', ncol=4, unormalize=True)
                self.vis.Images(images=vis_rgb, win='Current RGB', ncol=4, unormalize=True)
        
    def validate(self, e):
        rgb, thermal = next(iter(self.trainDataloader))
        rgb = rgb.to(self.device)
        thermal = thermal.to(self.device)
        
        with self.INV_ema.average_parameters():
            inv_z = self.INV(rgb)
        rec_images = self.G(thermal, inv_z)
        with self.D_ema.average_parameters():
            fake_labels, fake_rec_thermal = self.D(rec_images)
        vis_thermal, vis_rgb = self.draw_visualization(fake_rec_thermal=fake_rec_thermal, thermal=thermal,
                                                       fake_rgb=rec_images, rgb=rgb)
        self.vis.Images(images=vis_thermal, win="INV-Epoch-{}-thermal".format(e), ncol=4, unormalize=True)
        self.vis.Images(images=vis_rgb, win="INV-Epoch-{}-rgb".format(e), ncol=4, unormalize=True)
        
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
        if not os.path.isdir(self.args.saving_folder): os.makedirs(self.args.saving_folder)
        ckpt = {
            "D_parameter": self.D.state_dict(),
            "epoch": e,
            "D_optimizer": self.D_optimizer.state_dict(),
            "D_ema": self.D_ema.state_dict(),
            "INV_ema": self.INV_ema.state_dict(),
            "INV_parameter": self.INV.state_dict(),
            "INV_optimizer": self.INV_optimizer.state_dict()
        }
        torch.save(ckpt, '{}/checkpoint_inv.ckpt'.format(self.saving_folder))
        
        INV_ema_model = deepcopy(self.INV)
        D_ema_model = deepcopy(self.D)
        self.INV_ema.copy_to(INV_ema_model.parameters())
        self.D_ema.copy_to(D_ema_model.parameters())
        
        torch.save(INV_ema_model, '{}/INV.pt'.format(self.saving_folder))
        torch.save(D_ema_model, '{}/INV_D.pt'.format(self.saving_folder))
        
    def load(self):
        print("Loading G & D Model...")
        self.G = torch.load('{}/G.pt'.format(self.saving_folder))
        for p in self.G.parameters():
            p.requires_grad = False
        self.D = torch.load('{}/D.pt'.format(self.saving_folder))
        
def main():
    args = parse()
    vis = Visdom(args.env, args.port)
    Model = Model_factory(args, vis)
    
    if args.train:
        for e in range(1, args.epochs+1):
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