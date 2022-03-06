import torch
from torch import nn

norm = lambda in_c: nn.GroupNorm(num_groups=32, num_channels=in_c)


class ConvLSTMCellPeep(nn.Module):
    def __init__(self, in_size, in_c, hid_c, dropout_keep, kernel_size=3):
        super().__init__()
        self.height, self.width = in_size
        self.in_size = in_size
        self.in_c = in_c
        self.hid_c = hid_c
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_c + self.hid_c * 2, 2 * self.hid_c, kernel_size, 1, padding=kernel_size // 2),
            norm(2 * self.hid_c)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.in_c + self.hid_c * 2, self.hid_c, kernel_size, 1, padding=kernel_size // 2),
            norm(self.hid_c)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.in_c + self.hid_c, self.hid_c, kernel_size, 1, padding=kernel_size // 2),
            norm(self.hid_c)
        )
        self.dropout = nn.Dropout2d(p=1 - dropout_keep)  # dp2d instead of dropout for 2d feature map (Ma Xiao)
        print(f'ConvLSTMCell with peephole in_size = {in_size}, dropout {dropout_keep}')

    def forward(self, ins, seq_len, prev=None):
        if prev is None:
            h, c = self.init_states(ins.size(0))  # ins and prev not None at the same time (guaranteed)
        else:
            h, c = prev

        hs, cs = [], []  # store all intermediate h and c
        for i in range(seq_len):
            # prepare x: create one zero tensor if x is None (decoder mode)
            if ins is not None:
                x = ins[:, i]
            else:
                x = torch.zeros(h.size(0), self.in_c, self.height, self.width).cuda()

            x = self.dropout(x)  # conventional forward dropout
            h = self.dropout(h)  # variational inference based dropout (Gao, Y., et al. 2016)

            # f, i gates
            combined_conv = self.conv1(torch.cat([x, h, c], dim=1))

            i_t, f_t = torch.split(combined_conv, self.hid_c, dim=1)
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)

            # g gate
            g_t = self.conv3(torch.cat([x, h], dim=1))
            g_t = torch.tanh(g_t)

            # update cell state
            c_t = f_t * c + i_t * self.dropout(g_t)  # recurrent dropout (Semeniuta, S., et al., 2016)

            # o gate
            o_t = self.conv2(torch.cat([x, h, c_t], dim=1))
            o_t = torch.sigmoid(o_t)

            h_t = o_t * torch.tanh(c_t)

            h, c = h_t, c_t

            hs.append(h)
            cs.append(c)

        return hs, cs

    def init_states(self, batch_size):
        states = (torch.zeros(batch_size, self.hid_c, self.height, self.width),
                  torch.zeros(batch_size, self.hid_c, self.height, self.width))
        states = (states[0].cuda(), states[1].cuda())
        return states


class LSTMCellPeep(nn.Module):
    def __init__(self, in_c, hid_c, dropout_keep):
        super().__init__()
        self.in_c = in_c
        self.hid_c = hid_c
        self.fc1 = nn.Linear(self.in_c + self.hid_c * 2, 2 * self.hid_c)
        self.fc2 = nn.Linear(self.in_c + self.hid_c * 2, self.hid_c)
        self.fc3 = nn.Linear(self.in_c + self.hid_c, self.hid_c)
        self.dropout = nn.Dropout2d(p=1 - dropout_keep)  # dp2d instead of dropout for 2d feature map (Ma Xiao)
        print(f'ConvLSTMCell with peephole in_c = {in_c}, hidden_dim = {hid_c}, dropout {dropout_keep}')

    def forward(self, ins, seq_len, prev=None):
        if prev is None:
            h, c = self.init_states(ins.size(0))  # ins and prev not None at the same time (guaranteed)
        else:
            h, c = prev

        hs, cs = [], []  # store all intermediate h and c
        for i in range(seq_len):
            # prepare x: create one zero tensor if x is None (decoder mode)
            if ins is not None:
                x = ins[:, i]
            else:
                x = torch.zeros(h.size(0), self.in_c, self.height, self.width).cuda()

            x = self.dropout(x)  # conventional forward dropout
            h = self.dropout(h)  # variational inference based dropout (Gao, Y., et al. 2016)

            # f, i gates
            combined_conv = self.fc1(torch.cat([x, h, c], dim=1))

            i_t, f_t = torch.split(combined_conv, self.hid_c, dim=1)
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)

            # g gate
            g_t = self.fc3(torch.cat([x, h], dim=1))
            g_t = torch.tanh(g_t)

            # update cell state
            c_t = f_t * c + i_t * self.dropout(g_t)  # recurrent dropout (Semeniuta, S., et al., 2016)

            # o gate
            o_t = self.fc2(torch.cat([x, h, c_t], dim=1))
            o_t = torch.sigmoid(o_t)

            h_t = o_t * torch.tanh(c_t)

            h, c = h_t, c_t

            hs.append(h)
            cs.append(c)

        return hs, cs

    def init_states(self, batch_size):
        states = (torch.zeros(batch_size, self.hid_c),
                  torch.zeros(batch_size, self.hid_c))
        states = (states[0].cuda(), states[1].cuda())
        return states
