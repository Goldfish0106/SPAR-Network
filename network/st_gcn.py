import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .tgcn import ConvTemporalGraphical
from .graph import Graph

class STGCN_Model(nn.Module):
    def __init__(self, in_channels, out_channels, graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'}):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, out_channels, kernel_size, 1, residual=True),
            st_gcn(out_channels, out_channels, kernel_size, 1, residual=True),
            st_gcn(out_channels, out_channels, kernel_size, 1, residual=True),
        ))

    def forward(self, x):
        # forwad
        for gcn in self.st_gcn_networks:
            x, _ = gcn(x, self.A)
        return x


class st_gcn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A


if __name__ == "__main__":
    net = Model(in_channels=3, out_channels=64)
    x = torch.randn(32, 3, 300, 25)
    y = net(x)
    print(y.size())