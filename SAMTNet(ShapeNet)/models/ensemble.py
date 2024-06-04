import torch
import torch.nn as nn

class MetaModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MetaModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.linear(x)

class StackedEncoder(nn.Module):
    def __init__(self, encoder1, encoder2, meta_model, feature_size, height, width):
        super(StackedEncoder, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.meta_model = meta_model
        self.feature_size = feature_size
        self.height = height
        self.width = width

    def forward(self, rendering_images):
        features1 = self.encoder1(rendering_images)
        features2 = self.encoder2(rendering_images)
        # Flatten the features before concatenation
        features1_flat = features1.view(features1.size(0), -1)
        features2_flat = features2.view(features2.size(0), -1)
        stacked_features = torch.cat((features1_flat, features2_flat), dim=1)
        final_features = self.meta_model(stacked_features)
        # Reshape back to the original feature size
        batch_size = features1.size(0)
        final_features = final_features.view(batch_size, self.feature_size, self.height, self.width)
        return final_features.unsqueeze(1)  # Add a singleton dimension for depth
