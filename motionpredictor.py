import torch
from torch_geometric.nn import GATConv
from torch_geometric.utils import scatter
import config


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        input_dim = config.rotation_dim * config.context_length
        output_dim = config.rotation_dim * config.gen_frames
        root_pos_output_dim = 3 * config.gen_frames
        hid_lyrs = config.hid_lyrs
        heads_num = config.head_num
        dropout = config.dropout

        # Trunk layers: shared hidden layers
        trunk_dims = [input_dim] + hid_lyrs

        convs = []
        norms = []
        residual_projs = []
        current_dim = trunk_dims[0]

        for i, out_channels in enumerate(trunk_dims[1:]):
            convs.append(
                GATConv(
                    current_dim,
                    out_channels,
                    heads=heads_num,
                    add_self_loops=True,
                    dropout=dropout,
                )
            )

            next_dim = out_channels * heads_num

            if current_dim != next_dim:
                residual_projs.append(torch.nn.Linear(current_dim, next_dim))
            else:
                residual_projs.append(None)
            norms.append(torch.nn.LayerNorm(next_dim))

            current_dim = next_dim

        self.convs = torch.nn.ModuleList(convs)
        self.norms = torch.nn.ModuleList(norms)
        self.residual_projs = torch.nn.ModuleList(
            [p if p is not None else torch.nn.Identity() for p in residual_projs]
        )
        self._residual_is_identity = [p is None for p in residual_projs]
        self.activation = torch.nn.LeakyReLU()

        # Rotation head (per-joint output)
        self.rot_head = GATConv(
            current_dim,
            output_dim,
            heads=1,
            add_self_loops=True,
            dropout=dropout,
        )

        # Root position head (graph-level output via aggregation + MLP)
        root_pos_input_dim = current_dim + 3 * config.context_length
        self.root_pos_head = torch.nn.Sequential(
            torch.nn.Linear(root_pos_input_dim, current_dim // 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(current_dim // 2, root_pos_output_dim),
        )

        self.eps = 1e-8
        self.std_min = 1e-4
        self.value_clamp = 10.0

    def forward(self, src_graph):
        x = src_graph.x
        edge_index = src_graph.edge_index
        batch = src_graph.batch if hasattr(src_graph, 'batch') and src_graph.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Per-sample z-score normalization (scalar mean/std per graph)
        node_mean = x.mean(dim=1, keepdim=True)
        graph_mean = scatter(node_mean, batch, dim=0, reduce='mean')
        node_var = ((x - node_mean) ** 2).mean(dim=1, keepdim=True)
        graph_var = scatter(node_var, batch, dim=0, reduce='mean')
        graph_std = (graph_var + self.eps).sqrt()

        sample_mean = graph_mean[batch]
        sample_std = graph_std[batch].clamp(min=self.std_min)

        x = (x - sample_mean) / sample_std
        x = x.clamp(-self.value_clamp, self.value_clamp)

        # Shared trunk
        for i, conv in enumerate(self.convs):
            x_prev = x
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = x + self.residual_projs[i](x_prev)
            x = self.activation(x)

        # Rotation head (per-joint)
        rot_out = self.rot_head(x, edge_index)
        rot_out = rot_out.clamp(-self.value_clamp, self.value_clamp)
        rot_out = rot_out * sample_std + sample_mean

        # Root position head (graph-level)
        graph_feats = scatter(x, batch, dim=0, reduce='mean')  # (B, trunk_dim)

        # Get context root positions from input graph
        B = graph_feats.shape[0]
        root_pos_ctx = src_graph.root_pos.view(B, -1)  # (B, context*3)

        root_pos_input = torch.cat([graph_feats, root_pos_ctx], dim=-1)  # (B, trunk_dim + context*3)
        root_pos_out = self.root_pos_head(root_pos_input)  # (B, 3*gen_frames)

        return rot_out, root_pos_out

    @property
    def device(self):
        return next(self.parameters()).device
