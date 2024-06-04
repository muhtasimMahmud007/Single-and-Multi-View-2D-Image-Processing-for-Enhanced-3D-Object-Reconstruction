# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

import torch
import torchvision.models


import torch
import torchvision.models
 
 
class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg
 
        efficientNetB0 = torchvision.models.efficientnet_b0(pretrained=True)
        self.efficientnet = torch.nn.Sequential(*list(efficientNetB0.children())[:-2])
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1280, 512, kernel_size=3, padding=1),  # Adjust padding if needed
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(), #relu
            torch.nn.Upsample(size=(26,26), mode='bilinear',align_corners=True),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )
 
        for param in self.efficientnet.parameters():
            param.requires_grad = False
 
    def forward(self, rendering_images):
        # print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        image_features = []
 
        for img in rendering_images:
            features = self.efficientnet(img.squeeze(dim=0))
            #print(features.size())    # torch.Size([batch_size, 512, 28, 28])
            features = self.layer1(features)
            #print(features.size())    # torch.Size([batch_size, 512, 26, 26])
            features = self.layer2(features)
            #print(features.size())    # torch.Size([batch_size, 512, 8, 8])
            features = self.layer3(features)
            #print(features.size())    # torch.Size([batch_size, 256, 8, 8])
            image_features.append(features)
 
        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
        # print(image_features.size())  # torch.Size([batch_size, n_views, 256, 8, 8])
        return image_features