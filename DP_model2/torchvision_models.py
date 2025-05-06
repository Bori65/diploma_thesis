import torch
from torch import nn
from torchvision import models
from collections import OrderedDict
from einops.layers.torch import Rearrange

class ResNet18(nn.Module):
    def __init__(self, pretrained=False, convolutional_features=1024, out_features=256, dropout1=0.0, dropout2=0.0,
                 freeze=False):
        super().__init__()
        print("freeze: ", freeze)
        self.resnet = nn.Sequential(OrderedDict([
            ("conv1", models.resnet18(pretrained=pretrained, ).conv1),
            ("bn1", models.resnet18(pretrained=pretrained).bn1),
            ("relu", nn.ReLU(inplace=True)),
            ("maxpool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)),
            ("layer1", models.resnet18(pretrained=pretrained).layer1),
            ("layer2", models.resnet18(pretrained=pretrained).layer2),
            ("layer3", models.resnet18(pretrained=pretrained).layer3),
            ("layer4", models.resnet18(pretrained=pretrained).layer4),
            ("avgpool", nn.AdaptiveAvgPool2d(output_size=(1, 2 if convolutional_features == 1024 else 1))),
            ("flatten", Rearrange('b c w h -> b (c w h)')),
            ("dropout1", nn.Dropout(p=dropout1)),
            ("fc", nn.Linear(in_features=convolutional_features, out_features=out_features, bias=True)),
            ("dropout2", nn.Dropout(p=dropout2))
        ]))

        if freeze:
            for name, param in self.resnet.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

    def forward(self, x):
        return self.resnet(x)




if __name__ == "__main__":
    resnet18 = ResNet18(pretrained=True)
    print(resnet18)

    test = torch.zeros((1, 3, 224, 398))
    out = resnet18(test)
    print(out)
