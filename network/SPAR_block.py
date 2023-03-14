import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import Graph
from .attention import MultiHeadAttention
from .agcn import unit_tcn, TCN_GCN_unit


class PAR_ATT(nn.Module):
    def __init__(self, in_channels, out_channels, Nh, skel, stage, withSP, withTP, withSR, withTR):
        super().__init__()
        self.stage = stage
        self.dk = out_channels // Nh
        self.dv = out_channels // Nh
        self.out_channels = out_channels
        self.Wk = nn.Linear(in_channels, out_channels)
        self.Wq = nn.Linear(in_channels, out_channels)
        self.Wv = nn.Linear(in_channels, out_channels)
        self.att = MultiHeadAttention(d_model=in_channels, num_heads=Nh, drop_rate=0.)
        self.withTP = withTP

        if skel == 'NTU':
            if (not withSR) or stage == 0:
                if withSP:
                    self.partitions = ((3, 2, 20, 1, 0), (8, 9, 10, 11, 23, 24), (4, 5, 6, 7, 21, 22), (16, 17, 18, 19), (12, 13, 14, 15))
                else:
                    self.partitions = ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24),)
                self.vertex_num = 25
            elif stage == 1:
                if withSP:
                    self.partitions = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
                else:
                    self.partitions = ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9),)
                self.vertex_num = 10
            elif stage == 2:
                self.partitions = ((0, 1, 2, 3, 4),)
                self.vertex_num = 5
            else:
                raise AssertionError
            
            if (not withTR) or stage == 0:
                self.time_length = 300
                if withTP:
                    self.sample_num = 20
                else:
                    self.sample_num = 1
                self.sample_length = int(self.time_length / self.sample_num)
            elif stage == 1:
                self.time_length = 150
                if withTP:
                    self.sample_num = 15
                else:
                    self.sample_num = 1
                self.sample_length = int(self.time_length / self.sample_num)
            elif stage == 2:
                self.time_length = 75
                if withTP:
                    self.sample_num = 15
                else:
                    self.sample_num = 1
                self.sample_length = int(self.time_length / self.sample_num)
            else:
                raise AssertionError

    # input x: NM*C*T*V, 
    # where N is batch_size, 
    # M is the people num in one frame
    # C is the input channels
    # T is the total temporal length
    # V is the number of vertivals
    def forward(self, x, attn_embed):
        NM, C, T, S = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(NM*self.sample_num, self.sample_length, S, C)

        attn_bias = attn_embed(switched=True, stage=self.stage) if attn_embed is not None else None
        for partition in self.partitions:
            index = (np.array(partition).reshape((1,-1)) + np.arange(0, self.sample_length*self.vertex_num-1, self.vertex_num).reshape((-1,1))).flatten()
            part_attn_bias = attn_bias[index, index] if attn_bias is not None else 0

            part_x = x[:, :, partition, :].clone()
            N, T_part, V_part, Cin = part_x.size()
            part_x = part_x.view(N, T_part*V_part, Cin)

            Q = self.Wq(part_x)
            K = self.Wk(part_x)
            V = self.Wv(part_x)

            Q_ = torch.cat(torch.split(Q, self.dk, dim=-1), dim=0)
            K_ = torch.cat(torch.split(K, self.dk, dim=-1), dim=0)
            V_ = torch.cat(torch.split(V, self.dv, dim=-1), dim=0)
        
            part_x = self.cal_attention(Q_, K_, V_, part_attn_bias)
            part_x = torch.cat(torch.split(part_x, N, dim=0), dim=-1)
            # part_x, part_att = self.att(query=part_x, key=part_x, value=part_x, attn_bias=part_attn_bias)
            part_x = part_x.view(N, T_part, V_part, -1)
            x_clone = x.clone()
            x_clone[:,:,partition,:] = part_x
            x = x_clone

        # x = x.view(NM, self.sample_num, self.sample_length, S, self.out_channels)
        # x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(NM, T, S, self.out_channels)

        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def cal_attention(self, Q, K, V, attn_bias):
        K_T = torch.transpose(K, 1, 2)
        output = torch.matmul(Q, K_T) / np.sqrt(self.dk)
        output = output + attn_bias
        output = F.softmax(output, dim=-1)
        output = torch.matmul(output, V)
        return output


class SPAR_ATT(nn.Module):
    def __init__(self, in_channels, out_channels, Nh, skel, stage, withSP, withTP, withSR, withTR):
        super().__init__()
        self.stage = stage
        self.dk = out_channels // Nh
        self.dv = out_channels // Nh
        self.out_channels = out_channels
        self.Wk = nn.Linear(in_channels, out_channels)
        self.Wq = nn.Linear(in_channels, out_channels)
        self.Wv = nn.Linear(in_channels, out_channels)
        self.att = MultiHeadAttention(d_model=in_channels, num_heads=Nh, drop_rate=0.)

        if skel == 'NTU':
            if (not withSR) or stage == 0:
                if withSP:
                    self.partitions = ((8, 4, 20, 1, 0, 16, 12), (2, 9, 5, 17, 13), (3, 10, 6, 18, 14), (11, 7, 19, 15), (23, 24, 21, 22))
                else:
                    self.partitions = ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24),)
                self.vertex_num = 25
            elif stage == 1:
                if withSP:
                    self.partitions = ((2, 4, 1, 6, 8), (3, 5, 0, 7, 9))
                else:
                    self.partitions = ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9),)
                self.vertex_num = 10
            elif stage == 2:
                self.partitions = ((0, 1, 2, 3, 4),)
                self.vertex_num = 5
            else:
                raise AssertionError

            if (not withTR) or stage == 0:
                self.time_length = 300
                if withTP:
                    self.sample_num = 20
                else:
                    self.sample_num = 1
                self.sample_length = int(self.time_length / self.sample_num)
            elif stage == 1:
                self.time_length = 150
                if withTP:
                    self.sample_num = 15
                else:
                    self.sample_num = 1
                self.sample_length = int(self.time_length / self.sample_num)
            elif stage == 2:
                self.time_length = 75
                if withTP:
                    self.sample_num = 15
                else:
                    self.sample_num = 1
                self.sample_length = int(self.time_length / self.sample_num)
            else:
                raise AssertionError

    def forward(self, x, attn_embed):
        NM, C, T, S = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(NM, self.sample_length, self.sample_num, S, C)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(NM*self.sample_num, self.sample_length, S, C)

        attn_bias = attn_embed(switched=True, stage=self.stage) if attn_embed is not None else None
        for partition in self.partitions:
            index = (np.array(partition).reshape((1,-1)) + np.arange(0, self.sample_length*self.vertex_num-1, self.vertex_num).reshape((-1,1))).flatten()
            part_attn_bias = attn_bias[index, index] if attn_bias is not None else 0

            part_x = x[:, :, partition, :].clone()
            N, T_part, V_part, Cin = part_x.size()
            part_x = part_x.view(N, T_part*V_part, Cin)

            Q = self.Wq(part_x)
            K = self.Wk(part_x)
            V = self.Wv(part_x)

            Q_ = torch.cat(torch.split(Q, self.dk, dim=-1), dim=0)
            K_ = torch.cat(torch.split(K, self.dk, dim=-1), dim=0)
            V_ = torch.cat(torch.split(V, self.dv, dim=-1), dim=0)
        
            part_x = self.cal_attention(Q_, K_, V_, part_attn_bias)
            part_x = torch.cat(torch.split(part_x, N, dim=0), dim=-1)
            # part_x, part_att = self.att(query=part_x, key=part_x, value=part_x, attn_bias=part_attn_bias)
            part_x = part_x.view(N, T_part, V_part, -1)
            x_clone = x.clone()
            x_clone[:,:,partition,:] = part_x
            x = x_clone

        x = x.view(NM, self.sample_num, self.sample_length, S, self.out_channels)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(NM, T, S, self.out_channels)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def cal_attention(self, Q, K, V, attn_bias):
        K_T = torch.transpose(K, 1, 2)
        output = torch.matmul(Q, K_T) / np.sqrt(self.dk)
        output = output + attn_bias
        output = F.softmax(output, dim=-1)
        output = torch.matmul(output, V)
        return output


class PAR_Block(nn.Module):
    def __init__(self, in_channels, out_channels, Nh, skel, stage, withSP, withTP, withSR, withTR, T_kernel_size=9, stride=1, residual=True):
        super().__init__()
        self.par_att = PAR_ATT(in_channels, out_channels, Nh, skel=skel, stage=stage, withSP=withSP, withTP=withTP, withSR=withSR, withTR=withTR)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.T_kernel_size = T_kernel_size

        if T_kernel_size != 0:
            # padding = (T_kernel_size - 1) // 2
            # self.tcn = nn.Sequential(
            #     nn.Conv2d(
            #         out_channels,
            #         out_channels,
            #         (T_kernel_size, 1),
            #         (stride, 1),
            #         (padding, 0),
            #     ),
            #     nn.BatchNorm2d(out_channels),
            # )
            # self.relu2 = nn.ReLU(inplace=True)
            # self.tcn = unit_tcn(out_channels, out_channels, kernel_size=T_kernel_size, stride=stride)
            graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial', 'stage': stage}
            self.graph = Graph(**graph_args)
            A = self.graph.A
            self.tcn = TCN_GCN_unit(out_channels, out_channels, A, residual=False)

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

        

    def forward(self, x, attn_embed):
        res = self.residual(x)
        x = self.par_att(x, attn_embed)
        x = self.bn(x)
        x = self.relu1(x)
        # if self.T_kernel_size != 0:
        x = self.tcn(x)
        x = x + res
        # if self.T_kernel_size != 0:
        #     x = self.relu2(x)
        return x


class SPAR_Block(nn.Module):
    def __init__(self, in_channels, out_channels, Nh, skel, stage, withSP, withTP, withSR, withTR, T_kernel_size=9, stride=1, residual=True):
        super().__init__()
        self.spar_att = SPAR_ATT(in_channels, out_channels, Nh, skel=skel, stage=stage, withSP=withSP, withTP=withTP, withSR=withSR, withTR=withTR)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.T_kernel_size = T_kernel_size

        if T_kernel_size != 0:
            # padding = (T_kernel_size - 1) // 2
            # self.tcn = nn.Sequential(
            #     nn.Conv2d(
            #         out_channels,
            #         out_channels,
            #         (T_kernel_size, 1),
            #         (stride, 1),
            #         (padding, 0),
            #     ),
            #     nn.BatchNorm2d(out_channels),
            # )
            # self.relu2 = nn.ReLU(inplace=True)
            # self.tcn = unit_tcn(out_channels, out_channels, kernel_size=T_kernel_size, stride=stride)
            graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial', 'stage': stage}
            self.graph = Graph(**graph_args)
            A = self.graph.A
            self.tcn = TCN_GCN_unit(out_channels, out_channels, A, residual=False)

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

    

    def forward(self, x, attn_embed):
        res = self.residual(x)
        x = self.spar_att(x, attn_embed)
        x = self.bn(x)
        x = self.relu1(x)
        # if self.T_kernel_size != 0:
        x = self.tcn(x)
        x = x + res
        # if self.T_kernel_size != 0:
        #     x = self.relu2(x)
        return x
