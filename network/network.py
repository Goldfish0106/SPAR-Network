import torch
import torch.nn as nn
import torch.nn.functional as F

from .SPAR_block import PAR_Block, SPAR_Block
from .embedding import getSkeletonPE, AttentionEmbdding
# from .st_gcn import STGCN_Model
from .graph import Graph
from .agcn import TCN_GCN_unit

class Model(nn.Module):
    def __init__(self, in_channels, num_class, skel='NTU', base_dim=32, head_num=8, attn_drop=0., if_encode_attn=True, withSP=True, withTP=True, withSR=True, withTR=True, T_kernel_size=9):
        super().__init__()
        if skel == 'NTU':
            frame_len = 300
            frame_size = 25
            self.merge_list = [[2, 3], [20, 1, 0], [8, 9, 10], [11, 23, 24], [4, 5, 6], [7, 21, 22], [16, 17], [18, 19], [12, 13], [14, 15]]
            self.merge_proj = nn.Conv2d(in_channels=3*base_dim, out_channels=2*base_dim, kernel_size=(1, 1), bias=False)
        
        self.attn_embed = AttentionEmbdding(skel) if if_encode_attn else None    
        SkeletonPE = getSkeletonPE(time_len=frame_len, frame_size=frame_size, out_dim=base_dim)
        self.PositionalEncoding = nn.Parameter(SkeletonPE, requires_grad=False)
        # if not withSP:
        #     self.input_embed = nn.Conv2d(in_channels=in_channels, out_channels=base_dim, kernel_size=(1, 1), bias=False)
        #     nn.init.xavier_uniform_(self.input_embed.weight)
        #     self.stgcn = STGCN_Model(base_dim, base_dim)
        # else:
        #     self.input_embed = None
        # self.stgcn = STGCN_Model(in_channels, base_dim)
        graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'}
        self.graph = Graph(**graph_args)
        # A_ = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        A = self.graph.A
        # A = nn.Parameter(A_, requires_grad=False)

        self.stgcn = nn.Sequential(
            TCN_GCN_unit(in_channels, base_dim, A, residual=False),
            TCN_GCN_unit(base_dim, base_dim, A),
            TCN_GCN_unit(base_dim, base_dim, A)
        )

        self.spar_block0 = nn.ModuleList([
            PAR_Block(in_channels=base_dim, out_channels=base_dim, Nh=head_num, skel=skel, stage=0, withSP=withSP, withTP=withTP, withSR=withSR, withTR=withTR, T_kernel_size=T_kernel_size),
            SPAR_Block(in_channels=base_dim, out_channels=base_dim, Nh=head_num, skel=skel, stage=0, withSP=withSP, withTP=withTP, withSR=withSR, withTR=withTR, T_kernel_size=T_kernel_size),
        ])
        
        self.spar_block1 = nn.ModuleList([
            PAR_Block(in_channels=2*base_dim, out_channels=2*base_dim, Nh=head_num, skel=skel, stage=1, withSP=withSP, withTP=withTP, withSR=withSR, withTR=withTR, T_kernel_size=T_kernel_size),
            SPAR_Block(in_channels=2*base_dim, out_channels=2*base_dim, Nh=head_num, skel=skel, stage=1, withSP=withSP, withTP=withTP, withSR=withSR, withTR=withTR, T_kernel_size=T_kernel_size),
            PAR_Block(in_channels=2*base_dim, out_channels=2*base_dim, Nh=head_num, skel=skel, stage=1, withSP=withSP, withTP=withTP, withSR=withSR, withTR=withTR, T_kernel_size=T_kernel_size),
            SPAR_Block(in_channels=2*base_dim, out_channels=2*base_dim, Nh=head_num, skel=skel, stage=1, withSP=withSP, withTP=withTP, withSR=withSR, withTR=withTR, T_kernel_size=T_kernel_size),
        ])
        
        self.spar_block2 = nn.ModuleList([
            PAR_Block(in_channels=4*base_dim, out_channels=4*base_dim, Nh=head_num, skel=skel, stage=2, withSP=withSP, withTP=withTP, withSR=withSR, withTR=withTR, T_kernel_size=T_kernel_size),
            SPAR_Block(in_channels=4*base_dim, out_channels=4*base_dim, Nh=head_num, skel=skel, stage=2, withSP=withSP, withTP=withTP, withSR=withSR, withTR=withTR, T_kernel_size=T_kernel_size),
        ])
        if withSR and withTR:
            self.reduction1 = nn.Conv2d(in_channels=4*base_dim, out_channels=2*base_dim, kernel_size=(1, 1), bias=False)
            self.norm1 = nn.LayerNorm(2*base_dim)
            self.reduction2 = nn.Conv2d(in_channels=8*base_dim, out_channels=4*base_dim, kernel_size=(1, 1), bias=False)
            self.norm2 = nn.LayerNorm(4*base_dim)
        elif (not withSR) and (not withTR):
            self.reduction1 = nn.Conv2d(in_channels=base_dim, out_channels=2*base_dim, kernel_size=(1, 1), bias=False)
            self.reduction2 = nn.Conv2d(in_channels=2*base_dim, out_channels=4*base_dim, kernel_size=(1, 1), bias=False)
        else:
            self.reduction1 = nn.Conv2d(in_channels=2*base_dim, out_channels=2*base_dim, kernel_size=(1, 1), bias=False)
            self.reduction2 = nn.Conv2d(in_channels=4*base_dim, out_channels=4*base_dim, kernel_size=(1, 1), bias=False)

        nn.init.xavier_uniform_(self.reduction1.weight)
        nn.init.xavier_uniform_(self.reduction2.weight)
        self.fcn = nn.Conv2d(4*base_dim, num_class, kernel_size=1)
        self.withSR = withSR
        self.withTR = withTR

    def forward(self, x):
        N, C, T, V, M = x.size()
        # N, M, C, T, S
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.view(N*M, C, T, V)
        x = self.stgcn(x)
        x0 = x + self.PositionalEncoding
        for m in self.spar_block0:
            x0 = m(x0, self.attn_embed)
        if self.withSR:
            tensor_list = [x0[:, :, :, merge_index] for merge_index in self.merge_list]
            merged_tensor = list(map(self.merge, tensor_list))
            x1 = torch.stack(merged_tensor, dim=-1)
        else:
            x1 = x0
        if self.withTR:
            x1 = torch.cat((x1[:,:,0::2,:], x1[:,:,1::2,:]), dim=1)
        x1 = self.reduction1(x1)
        for m in self.spar_block1:
            x1 = m(x1, self.attn_embed)

        x2 = x1
        if self.withSR:
            x2 = torch.cat((x2[:,:,:,0::2], x2[:,:,:,1::2]), dim=1)
        if self.withTR:
            x2 = torch.cat((x2[:,:,0::2,:], x2[:,:,1::2,:]), dim=1)
        x2 = self.reduction2(x2)
        for m in self.spar_block2:
            x2 = m(x2, self.attn_embed)

        x2 = F.avg_pool2d(x2, x2.size()[2:])
        x2 = x2.view(N, M, -1, 1, 1).mean(dim=1)
        x2 = self.fcn(x2)
        x2 = x2.view(x2.size(0), -1)

        return x2

    def merge(self, x):
        N, C, T, V = x.size()
        if V == 2:
            return torch.cat((x[:, :, :, 0], x[:, :, :, 1]), dim=1)
        elif V == 3:
            cat_tensor = torch.cat((x[:, :, :, 0], x[:, :, :, 1], x[:, :, :, 2]), dim=1).unsqueeze(3)
            return self.merge_proj(cat_tensor).squeeze()
