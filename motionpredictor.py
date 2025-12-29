import torch
import math
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool
import config


class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # powers = torch.pow(1.25, torch.arange(1, max_len + 1, dtype=torch.float)).unsqueeze(1)
        # pe = powers.expand(max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, device):
        return self.pe.to(device)


class GATEncoder(torch.nn.Module):
    def __init__(self, z_dim):
        super(GATEncoder, self).__init__()

        self.context_length = config.context_length
        self.rotation_dim = config.rotation_dim
        self.bone_length_dim = config.bone_length_dim
        self.input_dim = self.bone_length_dim + self.rotation_dim * self.context_length
        self.dropout = config.dropout
        self.pe = config.pe

        if self.pe:
            self.rotation_pe = PositionalEncoding(max_len=self.context_length, d_model=self.rotation_dim)

        hid_lyrs = config.hid_lyrs
        heads_num = config.head_num

        e_Fs = [self.input_dim] + hid_lyrs + [z_dim]
        self.convs = []
        for i, (fi_prev, fi) in enumerate(zip(e_Fs[:-1], e_Fs[1:])):
            if i != 0:
                fi_prev *= heads_num
            if i != len(e_Fs) - 2:
                heads = heads_num
            else:
                heads = 1
            self.convs.append(
                GATConv(fi_prev, fi, heads=heads, add_self_loops=True, fill_value=0, dropout=self.dropout)
            )
        self.convs = torch.nn.ModuleList(self.convs)
        self.activation = torch.nn.LeakyReLU()

    def forward(self, src_graph):
        x = src_graph.x
        edge_index = src_graph.edge_index
        batch_id = src_graph.batch

        if self.pe:
            num_nodes = x.size(0)
            device = x.device
            dtype = x.dtype
            bone_len = x[:, : self.bone_length_dim]
            rotations = x[:, self.bone_length_dim:]
            rotations = rotations.view(num_nodes, self.context_length, self.rotation_dim)
            rot_pe = self.rotation_pe(device).to(dtype)  # (context_length, rotation_dim)
            rotations = rotations + rot_pe.unsqueeze(0)  # (num_nodes, context_length, rotation_dim)
            rotations = rotations.view(num_nodes, -1)
            x = torch.cat([bone_len, rotations], dim=-1)
            x = x.contiguous()

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            if (i + 1) != len(self.convs):
                x = self.activation(x)

        return global_max_pool(x, batch_id)


class GATDecoder(torch.nn.Module):
    def __init__(self, z_dim):
        super().__init__()

        rotation_dim = config.rotation_dim
        bone_length_dim = config.bone_length_dim
        gen_frames = config.gen_frames
        out_dim = rotation_dim * gen_frames
        dropout = config.dropout

        hid_lyrs = config.hid_lyrs
        heads_num = config.head_num
        tgt_all_lyr = config.tgt_all_lyr

        tgt_dim = bone_length_dim + rotation_dim

        d_Fs = [z_dim + tgt_dim] + hid_lyrs + [out_dim]
        self.deconvs = []

        for i, (fi_prev, fi) in enumerate(zip(d_Fs[:-1], d_Fs[1:])):
            heads = heads_num if i != len(d_Fs) - 2 else 1
            in_dim = fi_prev
            if i != 0:
                in_dim *= heads_num

            conv = GATConv(in_dim, fi, heads=heads, add_self_loops=True, dropout=dropout)
            self.deconvs.append(conv)

        self.deconvs = torch.nn.ModuleList(self.deconvs)
        self.activation = torch.nn.LeakyReLU()
        self.tgt_all_lyr = tgt_all_lyr

    def forward(self, src_z, tgt_graph):
        dec_x = src_z[tgt_graph.batch]
        tgt_x = tgt_graph.x

        edge_index = tgt_graph.edge_index
        dec_x = torch.hstack((dec_x, tgt_x))

        for i, conv in enumerate(self.deconvs):
            dec_x = conv(dec_x, edge_index)

            if (i + 1) != len(self.deconvs):
                dec_x = self.activation(dec_x)

        return dec_x


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        z_dim = config.z_dim

        self.encoder = GATEncoder(z_dim)
        self.decoder = GATDecoder(z_dim)

        self.z_dim = z_dim

    def forward(self, src_graph, lastframe_graph):
        z = self.encoder(src_graph)
        hatD = self.decoder(z, lastframe_graph)
        return hatD

    @property
    def device(self):
        return next(self.parameters()).device
