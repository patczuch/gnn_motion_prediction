import torch
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool
import config


class GATEncoder(torch.nn.Module):
    def __init__(self, z_dim):
        super(GATEncoder, self).__init__()

        context_length = config.context_length
        rotation_dim = config.rotation_dim
        bone_length_dim = config.bone_length_dim
        input_dim = bone_length_dim + rotation_dim * context_length

        hid_lyrs = config.hid_lyrs
        heads_num = config.head_num

        e_Fs = [input_dim] + hid_lyrs + [z_dim]
        self.convs = []
        for i, (fi_prev, fi) in enumerate(zip(e_Fs[:-1], e_Fs[1:])):
            if i != 0:
                fi_prev *= heads_num
            if i != len(e_Fs) - 2:
                heads = heads_num
            else:
                heads = 1
            self.convs.append(
                GATConv(fi_prev, fi, heads=heads, add_self_loops=True, fill_value=0)
            )
        self.convs = torch.nn.ModuleList(self.convs)

    def forward(self, src_graph):
        x = src_graph.x
        edge_index = src_graph.edge_index
        batch_id = src_graph.batch

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            if (i + 1) != len(self.convs):
                x = torch.nn.LeakyReLU()(x)

        return global_max_pool(x, batch_id)


class GATDecoder(torch.nn.Module):
    def __init__(self, z_dim):
        super().__init__()

        rotation_dim = config.rotation_dim
        bone_length_dim = config.bone_length_dim
        gen_frames = config.gen_frames
        out_dim = rotation_dim * gen_frames

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

            conv = GATConv(in_dim, fi, heads=heads, add_self_loops=True)
            self.deconvs.append(conv)

        self.deconvs = torch.nn.ModuleList(self.deconvs)
        self.tgt_all_lyr = tgt_all_lyr

    def forward(self, src_z, tgt_graph):
        dec_x = src_z[tgt_graph.batch]
        tgt_x = tgt_graph.x

        edge_index = tgt_graph.edge_index
        dec_x = torch.hstack((dec_x, tgt_x))

        for i, conv in enumerate(self.deconvs):
            dec_x = conv(dec_x, edge_index)

            if (i + 1) != len(self.deconvs):
                dec_x = torch.nn.LeakyReLU()(dec_x)

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
