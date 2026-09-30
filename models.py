import torch.nn as nn
import torch.nn.functional as F
import torch
import utils

"""VoxelMorph-style U-Net Convolution blocks"""
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)
    
"""VoxelMorph main model"""
class VoxelMorph(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(2,32,)
        self.enc2 = ConvBlock(32,64,)
        self.enc3 = ConvBlock(64,128,)

        # Decoder
        self.dec2 = ConvBlock(128 + 64,64,)
        self.dec1 = ConvBlock(64 + 32, 32,)

        # Predict displacement
        self.flow = nn.Conv2d(32,2,kernel_size=3,padding=1,)

        # Start with very small deformation
        nn.init.normal_( self.flow.weight,mean=0,std=1e-5,)
        nn.init.constant_(self.flow.bias, 0,)

    def forward(self,moving,fixed,):
        # Concatenate moving and fixed images
        x = torch.cat([moving, fixed],dim=1,)
        # Encoder
        e1 = self.enc1(x)
        p1 = F.avg_pool2d(e1,kernel_size=2,)
        e2 = self.enc2(p1)
        p2 = F.avg_pool2d(e2,kernel_size=2,)
        e3 = self.enc3(p2)

        # Decoder
        d2 = F.interpolate(e3,scale_factor=2,mode="bilinear",align_corners=True,)
        d2 = torch.cat([d2, e2],dim=1,)
        d2 = self.dec2(d2)
        d1 = F.interpolate(d2,scale_factor=2,mode="bilinear",align_corners=True,)
        d1 = torch.cat([d1, e1],dim=1,)
        d1 = self.dec1(d1)
        # Predict deformation field
        displacement = self.flow(d1)
        # Warp moving image
        warped = utils.warp_image(moving, displacement,)
        return warped, displacement