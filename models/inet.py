import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.tensor(np.eye(num_classes, dtype='float')[y.int().cpu()]).cuda().float().view(y.size(0), -1)


def convbn(in_channels, out_channels, kernel_size, stride, padding, bias):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class IntentionFeat(nn.Module):
    INTENT_FEAT_LEN = 64

    def __init__(self, num_intent, dropout):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_intent, self.INTENT_FEAT_LEN),
            nn.LeakyReLU(negative_slope=0.2, inplace=not True),
            nn.Dropout(p=1 - dropout, inplace=not True)
        )

    def forward(self, x):
        return self.fc(x)


class INet(nn.Module):
    NUM_VIEWS = 3
    VIEW_FEAT_LEN = [1000, 1000, 1000]
    FC_INTERM_LEN = 64

    def __init__(self, pretrained, fc_dropout_keep, intent_feat, num_modes, num_frames=1):
        super(INet, self).__init__()
        self.num_modes = num_modes

        self.intent_feat = intent_feat
        self.view_models = nn.ModuleList([resnet50(pretrained=pretrained) for i in range(self.NUM_VIEWS)])
        self.num_frames = num_frames

        if intent_feat:
            self.intent_fc = IntentionFeat(self.num_modes, fc_dropout_keep)

        fc_in = sum(self.VIEW_FEAT_LEN) + (self.intent_fc.INTENT_FEAT_LEN if intent_feat else 0)
        fc_in, self.FC_INTERM_LEN = fc_in * num_frames, self.FC_INTERM_LEN * int(num_frames ** 0.5)
        mlps = []
        for i in range(self.num_modes):
            mlps.append(
                nn.Sequential(
                    nn.Dropout(p=1 - fc_dropout_keep),
                    nn.Linear(fc_in, self.FC_INTERM_LEN, bias=True),
                    nn.Dropout(p=1 - fc_dropout_keep),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Linear(self.FC_INTERM_LEN, 2, bias=True)
                )
            )
        self.mlps = nn.ModuleList(mlps)

        print(f'model: assume 3-view inputs (bs, t, c, h, w), all intents are the same in a batch. '
              f'Num of intention = {self.num_modes}')

    def forward_feat(self, left, mid, right, intentions):
        # assume visual input (bs, c, h, w) or (bs, t, c, h, w). forward features for output head
        if len(left.shape) == 5:
            bs, t, c, h, w = left.shape
            assert t == self.num_frames, f'Expected {self.num_frames} frames as input, but got {t}'
        else:
            bs, c, h, w = left.shape
            t = 1

        views = [x.view(bs * t, c, h, w) for x in [left, mid, right]]
        intentions = intentions.view(bs * t)

        feats = []
        for i, view in enumerate(views):
            x = self.view_models[i](view)
            feats.append(x)
        x = torch.cat(feats, dim=1)  # (bs * t, c)

        if self.intent_feat:
            i_feat = self.intent_fc(to_categorical(intentions, self.num_modes))
            x = torch.cat((x, i_feat), dim=1)  # (bs * t, c)

        if t != 1:
            x = x.view(bs, t, x.size(1))  # (bs, t, c)

        return x

    def forward(self, left, mid, right, intentions):
        # assume input (bs, c, h, w) or (bs, t, c, h, w)
        # assume all intents are the same, or the intents at the last time step are the same

        feats = self.forward_feat(left, mid, right, intentions)

        if len(feats.shape) == 2:
            idx = int(intentions[0].item())
            x = self.mlps[idx](feats)
        else:
            idx = int(intentions[0][-1].item())
            x = self.mlps[idx](feats.view(feats.size(0), -1))

        return x


if __name__ == '__main__':
    t = 1
    model = INet(pretrained=False, fc_dropout_keep=0.7, intent_feat=False, num_modes=3,
                 num_frames=t).cuda()

    out = model(torch.randn(1, t, 3, 112, 112).cuda(),
                torch.randn(1, t, 3, 112, 112).cuda(),
                torch.randn(1, t, 3, 112, 112).cuda(),
                torch.tensor([[2] * t]).cuda())

    print(out.shape)
