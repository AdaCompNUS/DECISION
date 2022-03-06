import math

import torch
import torch.nn as nn

from models.convnet import SimpleConvNet
from models.lstm_cells import ConvLSTMCellPeep, LSTMCellPeep


class MemoryModule(nn.Module):
    def __init__(self, in_size, in_c, hid_c, dropout_keep):
        super().__init__()
        self.hid_c = hid_c
        self.cell = ConvLSTMCellPeep(in_size, in_c, hid_c, dropout_keep)

    def forward(self, x, prev):
        bs, t, c, h, w = x.shape
        x, cs = self.cell(x, t, prev=prev)  # prev could be None
        prev = (x[-1], cs[-1])  # only need the last dim
        x = torch.stack(x, dim=1)  # stack along t dim
        return x, prev


class ConvLSTMOneView(nn.Module):
    NUM_MODES = 4

    def __init__(self, spatial_size, channels, lstm_dropout_keep, sep_lstm, num_modes):
        super().__init__()
        self.spatial_size = spatial_size
        self.sep_lstm = sep_lstm

        if num_modes is not None:
            self.NUM_MODES = num_modes

        sizes = [(int(math.ceil(spatial_size[0] / 2 / (2 ** i))),
                  int(math.ceil(spatial_size[1] / 2 / (2 ** i)))) for i in range(1, 4)]

        self.conv_layers = self._get_conv_layers(channels)
        lstms = []
        for i in range(3):
            lstm_branches = []
            for j in range(self.NUM_MODES if sep_lstm else 1):
                cell = MemoryModule(sizes[i], channels[i], channels[i], lstm_dropout_keep)
                lstm_branches.append(cell)
            lstms.append(nn.ModuleList(lstm_branches))
        self.lstms = nn.ModuleList(lstms)

        self.pool_size = (2, 2)
        self.pool = nn.AdaptiveAvgPool2d(self.pool_size)

    def forward(self, x, prev, idx):
        prevs = []
        idx = idx if self.sep_lstm else 0
        convlstm_count = 0

        for i in range(3):
            bs, t, c, h, w = x.shape
            x = x.reshape(bs * t, c, h, w)
            x = self.conv_layers[i](x)
            _, c, h, w = x.shape
            x = x.reshape(bs, t, c, h, w)
            x, cs = self.lstms[convlstm_count][idx](x, prev[convlstm_count] if prev else None)
            convlstm_count += 1
            prevs.append(cs)

        x = self.pool(x.view(bs * t, c, h, w)).flatten(1).reshape(bs, t, c * self.pool_size[0] * self.pool_size[1])

        return x, prevs

    @staticmethod
    def _get_conv_layers(channels):
        return SimpleConvNet(channels).layers


class LSTMOneView(nn.Module):
    LSTM_DEPTH = 3

    def __init__(self, spatial_size, channels, lstm_dropout_keep, sep_lstm, num_modes):
        super().__init__()
        self.spatial_size = spatial_size
        self.sep_lstm = sep_lstm
        self.num_intention = num_modes

        self.conv_layers = self._get_conv_layers(channels)

        self.pool_size = (2, 2)
        self.pool = nn.AdaptiveAvgPool2d(self.pool_size)

        in_c = channels[-1] * self.pool_size[0] * self.pool_size[1]

        lstms = []
        for layer in range(self.LSTM_DEPTH):
            lstm_branches = nn.ModuleList([LSTMCellPeep(in_c=in_c, hid_c=in_c, dropout_keep=lstm_dropout_keep) for i in
                                           range(self.num_intention if sep_lstm else 1)])
            lstms.append(lstm_branches)
        self.lstms = nn.ModuleList(lstms)

    def _get_conv_layers(self, channels):
        return SimpleConvNet(channels).layers

    def forward(self, x, prev, idx):
        bs, t, c, h, w = x.shape
        x = x.view(bs * t, c, h, w)

        # propagate through 3-view feature extractors
        for i in range(3):
            x = self.conv_layers[i](x)

        # pool and flatten, got shape (bs, t, c)
        x = self.pool(x).flatten(1).view(bs, t, -1)

        # propagate through lstms
        idx = idx if self.sep_lstm else 0
        prevs = []
        for i in range(self.LSTM_DEPTH):
            x, cs = self.lstms[i][idx](x, t, prev[i] if prev else None)
            prevs.append((x[-1], cs[-1]))
            x = torch.stack(x, dim=1)  # stack along t dim

        return x, prevs


class DECISION(nn.Module):
    FC_DROPOUT_KEEP = 0.6
    LSTM_DROPOUT_KEEP = 0.7
    NUM_VIEWS = 4
    NAME_MODEL_MAPPING = {
        "decision": ConvLSTMOneView,
        "lstm": LSTMOneView,
    }

    def __init__(self, spatial_size, channels, sep_lstm, sep_fc, skip_depth, num_modes, controller_name='decision'):
        super(DECISION, self).__init__()
        self.sep_fc = sep_fc
        self.num_modes = num_modes

        views = []
        one_view_model = self.NAME_MODEL_MAPPING[controller_name]
        for i in range(self.NUM_VIEWS):
            views.append(
                one_view_model(spatial_size, channels, self.LSTM_DROPOUT_KEEP,
                               sep_lstm, num_modes)
            )
        self.view_models = nn.ModuleList(views)

        h, w = (2, 6)
        fc_in, fc_interm = channels[-1] * h * w, 2 * 32
        classifiers = []
        for i in range(self.num_modes if self.sep_fc else 1):
            classifiers.append(
                nn.Sequential(
                    nn.Dropout(p=1 - self.FC_DROPOUT_KEEP),
                    nn.Linear(fc_in, fc_interm, bias=True),
                    nn.Dropout(p=1 - self.FC_DROPOUT_KEEP),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Linear(fc_interm, 2, bias=True)
                )
            )
        self.classifiers = nn.ModuleList(classifiers)

        print(f'model: assume {self.NUM_VIEWS}-view inputs (bs, t, c, h, w), all mode signals in a batch are the same, '
              f'fc_dropout_keep {self.FC_DROPOUT_KEEP}, lstm_dropout_keep {self.LSTM_DROPOUT_KEEP}, '
              f'sep_lstm {sep_lstm}, skip_depth {skip_depth}, sep_fc = {sep_fc}, channels {channels}, '
              f'num of modes {self.num_modes}'
              f'\nWarning: Remember to manually reset/detach cell states during training!')

    def forward(self, left, mid, right, modes, prev):
        # assume input (bs, t, c, h, w) and all mode signals are the same
        idx = int(modes[0][0].item()) if self.sep_fc else 0
        views = [left, mid, right]

        feats, prevs = [], []
        for i, view in enumerate(views):
            x, cs = self.view_models[i % self.NUM_VIEWS](view, prev[i] if prev else None,
                                                         idx)  # 0 and 2 share the same view
            feats.append(x)
            prevs.append(cs)
        x = torch.cat(feats, dim=2)  # bs, t, c

        bs, t, c = x.shape
        x = x.view(bs * t, c)
        x = self.classifiers[idx](x)
        x = x.view(bs, t, -1)

        return x, prevs

    @staticmethod
    def detach_states(states):
        for depth_states in states:
            for i, (h, c) in enumerate(depth_states):
                h, c = h.detach(), c.detach()
                h.requires_grad, c.requires_grad = True, True
                depth_states[i] = (h, c)
        return states

    @staticmethod
    def derive_grad(y, x):
        for depth_y, depth_x in zip(y, x):
            for (yh, yc), (xh, xc) in zip(depth_y, depth_x):
                yc.backward(xc.grad, retain_graph=False)  # False still in testing


if __name__ == '__main__':
    model = DECISION((112, 112), [128, 192, 256], sep_lstm=True, sep_fc=True, skip_depth=[], nviews=3,
                     num_modes=3, controller_name='lstm').cuda()
    out = model(torch.randn(1, 4, 3, 112, 112).cuda(),
                torch.randn(1, 4, 3, 112, 112).cuda(),
                torch.randn(1, 4, 3, 112, 112).cuda(),
                torch.randn(1, 2).cuda(),
                None)

    print(out[0].shape, len(out[1]), len(out[1][0]))
