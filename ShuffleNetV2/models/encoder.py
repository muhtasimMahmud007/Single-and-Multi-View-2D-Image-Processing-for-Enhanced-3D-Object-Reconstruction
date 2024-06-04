import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        # Load ShuffleNetV2 model
        shufflenet_model = models.shufflenet_v2_x1_0(pretrained=True)
        
        # Extract the feature layers
        self.features = nn.Sequential(*list(shufflenet_model.children())[:-1])

        # Additional layers to match decoder input requirements
        self.layer1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU()
        )

    def forward(self, rendering_images):
        # print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        image_features = []

        for img in rendering_images:
            features = self.features(img.squeeze(dim=0))
            # print(features.size())    # torch.Size([batch_size, 1024, 7, 7])
            features = self.layer1(features)
            # print(features.size())    # torch.Size([batch_size, 512, 7, 7])
            features = self.layer2(features)
            # print(features.size())    # torch.Size([batch_size, 512, 7, 7])
            features = self.layer3(features)
            # print(features.size())    # torch.Size([batch_size, 1568, 7, 7])
            image_features.append(features)

        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
        # print(image_features.size())  # torch.Size([batch_size, n_views, 1568, 7, 7])
        return image_features