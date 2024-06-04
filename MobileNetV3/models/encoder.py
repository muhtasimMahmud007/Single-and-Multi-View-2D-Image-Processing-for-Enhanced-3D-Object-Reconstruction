import torch
import torchvision.models


class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        mobilenet_v3 = torchvision.models.mobilenet_v3_small(pretrained=True)
        self.mobilenet = mobilenet_v3.features
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(576, 512, kernel_size=3),  # Adjusting input channels to match MobileNetV3 output
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
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

        # Don't update params in MobileNetV3
        for param in mobilenet_v3.parameters():
            param.requires_grad = False

    def forward(self, rendering_images):
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        image_features = []

        for img in rendering_images:
            features = self.mobilenet(img.squeeze(dim=0))
            #print(features.size())
            features = self.layer1(features)
            features = self.layer2(features)
            features = self.layer3(features)
            image_features.append(features)

        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
        #print(image_features.size()) 
        return image_features





