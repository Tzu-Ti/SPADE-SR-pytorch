import torch
import torchvision
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF

import numpy as np

class TherDataset(Dataset):
    def __init__(self, rgb_npy_path, thermal_npy_path, augment=True):
        self.augment = augment
        self.rgb_npys = np.load(rgb_npy_path)
        self.thermal_npys = np.load(thermal_npy_path)
        
        # numbers of rgb images must equal to thermal images
        assert self.rgb_npys.shape[0] == self.thermal_npys.shape[0]
        
        # transforms
        self.transforms_rgb_aug = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transforms_rgb = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transforms_thermal = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5), (0.5))
        ])
        
    def __getitem__(self, index):
        rgb = self.rgb_npys[index]
        thermal = self.thermal_npys[index]
        
        if self.augment:
            rgb = self.transforms_rgb_aug(rgb)
            thermal = self.transforms_thermal(thermal)
            if torch.rand(1) > 0.5:
                rgb = TF.hflip(rgb)
                thermal = TF.hflip(thermal)
            return rgb, thermal
                
        rgb = self.transforms_rgb(rgb)
        thermal = self.transforms_thermal(thermal)
        
        return rgb, thermal
        
    def __len__(self):
        return self.rgb_npys.shape[0]