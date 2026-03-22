import torch
from torch_geometric.nn import GATConv
from torch_geometric.utils import scatter
import config


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        input_dim = config.rotation_dim * config.context_length
        output_dim = config.rotation_dim * config.gen_frames
        hid_lyrs = config.hid_lyrs
        heads_num = config.head_num
        dropout = config.dropout

        layer_dims = [input_dim] + hid_lyrs + [output_dim]

        convs = []
        norms = []
        current_dim = layer_dims[0]

        for i, out_channels in enumerate(layer_dims[1:]):
            is_last = i == len(layer_dims[1:]) - 1
            heads = 1 if is_last else heads_num

            convs.append(
                GATConv(
                    current_dim,
                    out_channels,
                    heads=heads,
                    add_self_loops=True,
                    dropout=dropout,
                )
            )

            current_dim = out_channels * heads

            if not is_last:
                norms.append(torch.nn.LayerNorm(current_dim))

        self.convs = torch.nn.ModuleList(convs)
        self.norms = torch.nn.ModuleList(norms)
        self.activation = torch.nn.LeakyReLU()
        self.eps = 1e-8
        self.std_min = 1e-4
        self.value_clamp = 10.0

    def forward(self, src_graph):
        x = src_graph.x
        edge_index = src_graph.edge_index
        batch = src_graph.batch if hasattr(src_graph, 'batch') and src_graph.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Per-sample z-score normalization (scalar mean/std per graph)
        node_mean = x.mean(dim=1, keepdim=True)  # (N, 1)
        graph_mean = scatter(node_mean, batch, dim=0, reduce='mean')  # (B, 1)
        node_var = ((x - node_mean) ** 2).mean(dim=1, keepdim=True)  # (N, 1)
        graph_var = scatter(node_var, batch, dim=0, reduce='mean')  # (B, 1)
        graph_std = (graph_var + self.eps).sqrt()  # (B, 1)

        sample_mean = graph_mean[batch]  # (N, 1)
        sample_std = graph_std[batch].clamp(min=self.std_min)    # (N, 1)

        x = (x - sample_mean) / sample_std
        x = x.clamp(-self.value_clamp, self.value_clamp)

        norm_idx = 0
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i + 1 != len(self.convs):
                x = self.norms[norm_idx](x)
                norm_idx += 1
                x = self.activation(x)

        # Clamp output to prevent explosion during denormalization
        x = x.clamp(-self.value_clamp, self.value_clamp)

        # Denormalize output
        x = x * sample_std + sample_mean

        return x

    @property
    def device(self):
        return next(self.parameters()).device
