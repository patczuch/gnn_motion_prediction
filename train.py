import torch
from torch import nn
from torch.utils.data import DataLoader
from motionpredictor import MotionPredictor
from bvh_utils import load_motion_sequences, get_bone_hierarchy
from dataset import MotionDataset


def train(model, dataloader, edge_index, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(50):
        total_loss = 0
        for seq, target in dataloader:
            seq = seq.to(device)
            target = target.to(device)

            pred = model(seq, edge_index)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss = {total_loss / len(dataloader):.6f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sequences = load_motion_sequences()
    sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]

    dataset = MotionDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_joints = sequences[0].shape[1]
    feature_dim = sequences[0].shape[2]

    bone_edges = get_bone_hierarchy()

    edge_index = torch.tensor(bone_edges, dtype=torch.long).t().contiguous().to(device)

    model = MotionPredictor(feature_dim, hidden_dim=64, num_joints=num_joints, out_dim=num_joints * 9).to(device)

    train(model, dataloader, edge_index, device)
