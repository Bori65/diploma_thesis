import torch
from torch import nn
from einops import rearrange
from collections import OrderedDict
from torchesn.nn import ESN
from torchvision_models import ResNet18


class CompositModel(nn.Module):
    def __init__(self, vision_architecture, pretrained_vision, seq2seq_architecture, dropout1=0.5, dropout2=0.0,
                 image_features=256, hidden_dim=512, freeze=False, precooked=False, convolutional_features=1024,
                 no_joints=False, leaking_rate = 0.02, lambda_reg=0.01, spectral_radius=0.98, output_steps='last', readout_training='gd'):
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

        if seq2seq_architecture == "esn_pytorch":
            self.esn_model = ESN(input_size=image_features + JOINTS_SIZE, hidden_size=hidden_dim, output_size=LABEL_SIZE,
                                batch_first = True, leaking_rate = leaking_rate, lambda_reg=lambda_reg,
                                 spectral_radius=spectral_radius, output_steps=output_steps, readout_training=readout_training)
        else:
            raise ValueError("Wrong seq2seq model!")

    def forward(self, frames, joints, labels, test):
        # input:
        # images shape : (N, L, 3, 224, 398)
        #   if precooked -> (N, L, conv_features)
        # joints shape : (N, L, 6)
        #
        # ouput:
        # output shape : (N, Lout), Lout = 3

        N = frames.shape[0]  # batch size
        L = frames.shape[1]  # sequence length
        Lout = 3  # length of output sequence
        t = 19  # token size

        # forward pass
        if self.precooked:
            frames = rearrange(frames, 'N L cf -> (N L) cf')
        else:
            frames = rearrange(frames, 'N L c w h -> (N L) c w h')

        frames_features = self.vision_model(frames)  # shape : (N L) feature_dim

        frames_features = rearrange(frames_features, '(N L) f -> N L f', N=N, L=L)

        if self.no_joints:
            sequence_input = frames_features  # shape : (N, L, f)
        else:
            sequence_input = torch.cat((frames_features, joints), dim=2)  # shape : (N, L, f+j)
        #washout = torch.full((sequence_input.shape[0],), 15, device=sequence_input.device, dtype=torch.float32)
        if not test:
            washout = [15]*sequence_input.shape[0]
        elif test:
            washout = [0] * sequence_input.shape[0]

        output, hidden = self.esn_model(sequence_input, washout)
        output = output.squeeze(1)
        print("test:", test, "output shape: ", output.shape)

        return output
