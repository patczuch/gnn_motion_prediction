import torch
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool


class GATEncoder(torch.nn.Module):
    def __init__(self, z_dim):
        super(GATEncoder, self).__init__()

        context_length = 10
        rotation_dim = 9
        input_dim = rotation_dim * context_length

        hid_lyrs = [16, 16, 16]
        heads_num = 16

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
        x = src_graph.src_x
        edge_index_bi = src_graph.edge_index_bidirection
        batch_id = src_graph.batch

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index_bi)

            if (i + 1) != len(self.convs):
                x = torch.nn.ReLU()(x)

        V_mask = src_graph.mask
        if V_mask.sum() > 0:
            pool_z_x = global_max_pool(x[~V_mask], batch_id[~V_mask])
        else:
            pool_z_x = global_max_pool(x, batch_id)

        return pool_z_x


class GATDecoder(torch.nn.Module):
    def __init__(self, z_dim):
        super().__init__()

        out_dim = 9

        hid_lyrs = [16, 16, 16]
        heads_num = 16
        tgt_all_lyr = True

        tgt_dim = 9
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
        tgt_x = tgt_graph.tgt_x

        edge_index_bi = tgt_graph.edge_index_bidirection
        dec_x = torch.hstack((dec_x, tgt_x))

        for i, conv in enumerate(self.deconvs):
            dec_x = conv(dec_x, edge_index_bi)

            if (i + 1) != len(self.deconvs):
                dec_x = torch.nn.ReLU()(dec_x)

        return dec_x


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        z_dim = 32

        self.encoder = GATEncoder(z_dim)
        self.decoder = GATDecoder(z_dim)

        self.z_dim = z_dim

    def forward(self, src_graph, tgt_graph):
        z = self.encoder(src_graph)
        hatD = self.decoder(z, tgt_graph)
        return z, hatD

    @property
    def device(self):
        return next(self.parameters()).device
