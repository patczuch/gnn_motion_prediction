import torch
from torch import nn
from torch_geometric.loader import DataLoader
from motionpredictor import Model
from dataset import BVHMotionDataset


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = BVHMotionDataset("./datasets/lafan1test", context=10)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Model().to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(50):
        epoch_loss = 0

        for src_graph, tgt_graph in dataloader:
            src_graph = src_graph.to(device)
            tgt_graph = tgt_graph.to(device)

            z, hatD = model(src_graph, tgt_graph)

            gt = tgt_graph.tgt_x.to(device)

            loss = loss_fn(hatD, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss = {epoch_loss / len(dataloader):.6f}")
