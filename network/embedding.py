import torch
import torch.nn as nn
import numpy as np

from .utils.graph import Graph

def getPositionEncoding(seq_len, out_dim, n=10000):
    P = torch.zeros(seq_len, out_dim)
    for k in range(seq_len):
        for i in np.arange(int(out_dim/2)):
            denominator = np.power(n, 2*i/out_dim)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P

def getSkeletonPE(time_len, frame_size, out_dim):
    P = getPositionEncoding(time_len*frame_size, out_dim)
    SkeletonPE = P.transpose(0,1).view(out_dim, time_len, frame_size)
    return SkeletonPE

def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class AttentionEmbdding(nn.Module):
    def __init__(self, skel='NTU'):
        super().__init__()
        self.graph = Graph(skel)
        if skel == 'NTU':
            self.max_par_dist = 14
            self.max_spar_dist = 42
            self.gragh = Graph(skel='NTU')
            self.dist_par_0 = nn.Parameter(torch.zeros(375, 375).int(), requires_grad=False)
            self.dist_par_1 = nn.Parameter(torch.zeros(100, 100).int(), requires_grad=False)
            self.dist_par_2 = nn.Parameter(torch.zeros(25, 25).int(), requires_grad=False)
            self.dist_spar_0 = nn.Parameter(torch.zeros(375, 375).int(), requires_grad=False)
            self.dist_spar_1 = nn.Parameter(torch.zeros(100, 100).int(), requires_grad=False)
            self.dist_spar_2 = nn.Parameter(torch.zeros(25, 25).int(), requires_grad=False)
        self.init_dist(skel)

        self.par_encoder0 = nn.Embedding(self.max_par_dist, 1)
        self.spar_encoder0 = nn.Embedding(self.max_spar_dist, 1)
        self.par_encoder1 = nn.Embedding(self.max_par_dist, 1)
        self.spar_encoder1 = nn.Embedding(self.max_spar_dist, 1)
        self.par_encoder2 = nn.Embedding(self.max_par_dist, 1)
        self.spar_encoder2 = nn.Embedding(self.max_spar_dist, 1)
        self.apply(lambda module: init_params(module))

    def forward(self, switched, stage):
        if not switched:
            if stage == 0:
                return self.par_encoder0(self.dist_par_0).squeeze()
            elif stage == 1:
                return self.par_encoder1(self.dist_par_1).squeeze()
            elif stage == 2:
                return self.par_encoder2(self.dist_par_2).squeeze()
            else:
                raise AssertionError

        else:
            if stage == 0:
                return self.spar_encoder0(self.dist_spar_0).squeeze()
            elif stage == 1:
                return self.spar_encoder1(self.dist_spar_1).squeeze()
            elif stage == 2:
                return self.spar_encoder2(self.dist_spar_2).squeeze()
            else:
                raise AssertionError


    def init_dist(self, skel):
        if skel == 'NTU':
            for i in range(self.dist_par_0.size()[0]):
                for j in range(self.dist_par_0.size()[1]):
                    t1 = i // 25
                    v1 = i % 25
                    t2 = j // 25
                    v2 = j % 25
                    self.dist_par_0[i][j] = self.graph.minDist0[v1][v2] + int(abs(t1-t2) / 10)

            for i in range(self.dist_par_1.size()[0]):
                for j in range(self.dist_par_1.size()[1]):
                    t1 = i // 10
                    v1 = i % 10
                    t2 = j // 10
                    v2 = j % 10
                    self.dist_par_1[i][j] = self.graph.minDist1[v1][v2] + int(abs(t1-t2) / 10)

            for i in range(self.dist_par_2.size()[0]):
                for j in range(self.dist_par_2.size()[1]):
                    t1 = i // 5
                    v1 = i % 5
                    t2 = j // 5
                    v2 = j % 5
                    self.dist_par_2[i][j] = self.graph.minDist2[v1][v2] + int(abs(t1-t2) / 10)

            for i in range(self.dist_spar_0.size()[0]):
                for j in range(self.dist_spar_0.size()[1]):
                    t1 = i // 25
                    v1 = i % 25
                    t2 = j // 25
                    v2 = j % 25
                    self.dist_spar_0[i][j] = self.graph.minDist0[v1][v2] + int(abs(t1-t2) * 2)

            for i in range(self.dist_spar_1.size()[0]):
                for j in range(self.dist_spar_1.size()[1]):
                    t1 = i // 10
                    v1 = i % 10
                    t2 = j // 10
                    v2 = j % 10
                    self.dist_spar_1[i][j] = self.graph.minDist1[v1][v2] + int(abs(t1-t2) * 1.5)

            for i in range(self.dist_spar_2.size()[0]):
                for j in range(self.dist_spar_2.size()[1]):
                    t1 = i // 5
                    v1 = i % 5
                    t2 = j // 5
                    v2 = j % 5
                    self.dist_spar_2[i][j] = self.graph.minDist2[v1][v2] + int(abs(t1-t2) * 1.5)


if __name__ == "__main__":
    SkePE = getSkeletonPE(300, 25, 64)
    print(SkePE.size())