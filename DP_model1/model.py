import numpy as np
import torch
from torch import nn
from einops import rearrange
from collections import OrderedDict
from torchvision_models import ResNet34, ResNet50, ResNet101, ResNet18, EisermannVGG
from reservoirpy.nodes import Reservoir, Ridge


class VisionModel(nn.Module):
    def __init__(self, vision_architecture, pretrained_vision, dropout1=0.0, dropout2=0.0,
                 image_features=256, freeze=False, precooked=False, convolutional_features=1024,
                 no_joints=False):
        super().__init__()

        LABEL_SIZE = 19
        JOINTS_SIZE = 0 if no_joints else 6
        self.precooked = precooked
        self.no_joints = no_joints

        if self.precooked:
            self.vision_model = nn.Sequential(OrderedDict([
                ("dropout1", nn.Dropout(p=dropout1)),
                ("fc", nn.Linear(in_features=convolutional_features, out_features=image_features, bias=True)),
                ("dropout2", nn.Dropout(p=dropout2))
            ]))
        else:
            if vision_architecture == "resnet18":
                print("pretraine: ",pretrained_vision)
                self.vision_model = ResNet18(pretrained=pretrained_vision,
                                             convolutional_features=convolutional_features, out_features=image_features,
                                             dropout1=dropout1, dropout2=dropout2, freeze=freeze)
            elif vision_architecture == "resnet34":
                self.vision_model = ResNet34(pretrained=pretrained_vision,
                                             convolutional_features=convolutional_features, out_features=image_features,
                                             dropout1=dropout1, dropout2=dropout2, freeze=freeze)
            elif vision_architecture == "resnet50":
                self.vision_model = ResNet50(pretrained=pretrained_vision,
                                             convolutional_features=convolutional_features, out_features=image_features,
                                             dropout1=dropout1, dropout2=dropout2, freeze=freeze)
            elif vision_architecture == "resnet101":
                self.vision_model = ResNet101(pretrained=pretrained_vision,
                                              convolutional_features=convolutional_features,
                                              out_features=image_features,
                                              dropout1=dropout1, dropout2=dropout2, freeze=freeze)
            elif vision_architecture == "eisermann_vgg":
                self.vision_model = EisermannVGG(out_features=image_features, dropout2=dropout2, freeze=freeze)
            else:
                raise ValueError("Wrong vision model!")



    def forward(self, frames, joints):
        N = frames.shape[0]  # batch size
        L = frames.shape[1]  # sequence length

        # Forward pass through vision model
        if self.precooked:
            frames = rearrange(frames, 'N L cf -> (N L) cf')
        else:
            frames = rearrange(frames, 'N L c w h -> (N L) c w h')

        frames_features = self.vision_model(frames)
        frames_features = rearrange(frames_features, '(N L) f -> N L f', N=N, L=L)

        # Concatenate joints if available
        sequence_input = frames_features if self.no_joints else torch.cat((frames_features, joints), dim=2)

        output = sequence_input.detach().cpu().numpy()
        return output
