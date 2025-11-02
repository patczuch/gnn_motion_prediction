import os
import torch
import time
from torch import nn
from torch_geometric.loader import DataLoader
from motionpredictor import Model
from dataset import BVHMotionDataset


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataset = BVHMotionDataset("./datasets/lafan1train", context=10, step=10)

    val_ratio = 0.2
    total = len(dataset)
    val_size = int(total * val_ratio)
    train_size = total - val_size

    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size = 32

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)

    model = Model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss(reduction='sum')

    print(f"Train size: {len(train_loader) * batch_size}, Validation size: {len(val_loader) * batch_size}")

    for epoch in range(50):
        model.train()
        train_loss = 0.0

        for src_graph, tgt_graph in train_loader:
            src_graph = src_graph.to(device)
            tgt_graph = tgt_graph.to(device)

            _, pred = model(src_graph, tgt_graph)
            gt = tgt_graph.x

            loss = loss_fn(pred, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src_graph, tgt_graph in val_loader:
                src_graph = src_graph.to(device)
                tgt_graph = tgt_graph.to(device)

                _, pred = model(src_graph, tgt_graph)
                gt = tgt_graph.x.to(device)

                loss = loss_fn(pred, gt)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch + 1:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model_" + time.strftime("%Y%m%d-%H%M%S") + ".pth"))
