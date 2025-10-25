import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


def batch_edge_index(edge_index, num_nodes, batch_size):
    edge_index_list = []
    for i in range(batch_size):
        edge_index_list.append(edge_index + i * num_nodes)
    return torch.cat(edge_index_list, dim=1)


class GATFrameEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=2)
        self.gat2 = GATConv(hidden_dim*2, hidden_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index).relu()
        x = self.gat2(x, edge_index)
        return x  # shape: (num_joints, hidden_dim)


class MotionPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_joints):
        super().__init__()
        self.encoder = GATFrameEncoder(in_dim, hidden_dim)

        # GRU processes sequence of graph embeddings (one embedding per joint)
        self.gru = nn.GRU(hidden_dim*num_joints, hidden_dim, batch_first=True)

        # Decoder predicts next joint features
        self.decoder = GATFrameEncoder(hidden_dim, out_dim)

        self.num_joints = num_joints
        self.hidden_dim = hidden_dim

    def forward(self, seq, edge_index):
        T = seq.shape[0]
        num_joints = seq.shape[1]

        # Flatten sequence for batch processing
        x_batch = seq.reshape(T * num_joints, -1)
        edge_index_batch = batch_edge_index(edge_index, num_joints, T)

        # Encode all frames in one pass
        enc = self.encoder(x_batch, edge_index_batch)  # (T*num_joints, hidden_dim)
        enc_frames = enc.view(T, num_joints, -1)

        # Flatten joints for GRU
        enc_frames = enc_frames.reshape(1, T, -1)

        _, h = self.gru(enc_frames)  # (1, 1, hidden_dim)

        h = h.squeeze(0).unsqueeze(0).repeat(num_joints, 1)
        out = self.decoder(h, edge_index)
        return out