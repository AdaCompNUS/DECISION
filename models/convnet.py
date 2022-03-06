from torch import nn as nn


def convbn(in_channels, out_channels, kernel_size, stride, padding, bias):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class SimpleConvNet(nn.Module):
    CHANNELS = [3, 64]

    def __init__(self, channels):
        super().__init__()
        layer1 = nn.Sequential(
            convbn(self.CHANNELS[0], self.CHANNELS[1], kernel_size=7, stride=2, padding=3, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            convbn(self.CHANNELS[1], channels[0], kernel_size=3, stride=1, padding=1, bias=True)
        )
        layer2 = convbn(channels[0], channels[1], kernel_size=3, stride=2, padding=1, bias=True)
        layer3 = convbn(channels[1], channels[2], kernel_size=3, stride=2, padding=1, bias=True)
        self.layers = nn.ModuleList([layer1, layer2, layer3])

    def forward(self, x):
        return self.layers(x)
