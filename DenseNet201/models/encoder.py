import torch
import torchvision.models


class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        dense201 = torchvision.models.densenet201(weights=torchvision.models.DenseNet201_Weights.DEFAULT)
        self.densenet = dense201.features

        for param in self.densenet.parameters():
            param.requires_grad = False

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(dense201.classifier.in_features, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Upsample(size=(8,8), mode='bilinear',align_corners=True),
        )

    def forward(self, rendering_images):
        # print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        image_features = []

        for img in rendering_images:
            features = self.densenet(img.squeeze(dim=0))
            # print(features.size())    # torch.Size([batch_size, 1920, 7, 7])
            features = self.layer1(features)
            # print(features.size())    # torch.Size([batch_size, 512, 7, 7])
            features = self.layer2(features)
            # print(features.size())    # torch.Size([batch_size, 256, 8, 8])
            image_features.append(features)

        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
        # print(image_features.size())  # torch.Size([batch_size, n_views, 256, 8, 8])
        return image_features