import os
import torch
import time

from torch_geometric.loader import DataLoader

from geodesicloss import GeodesicLoss
from motionpredictor import Model
from dataset import BVHMotionDataset
import pymotion.rotations.ortho6d_torch as sixd_torch


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    context = 20

    dataset = BVHMotionDataset("./datasets/lafan1train_small", context=context, step=context)

    val_ratio = 0.2
    total = len(dataset)
    val_size = int(total * val_ratio)
    train_size = total - val_size

    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size = 128

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)

    model = Model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = GeodesicLoss(reduction='mean')

    print(f"Train size: {len(train_loader) * batch_size}, Validation size: {len(val_loader) * batch_size}")
    rotation_dim = 6
    rollout = 5

    for epoch in range(5):
        model.train()
        train_loss = 0.0

        for src_graph, lastframe_graph, tgt_graph in train_loader:
            src_graph = src_graph.to(device)
            lastframe_graph = lastframe_graph.to(device)
            tgt_graph = tgt_graph.to(device)

            J = src_graph.x.shape[0]
            context_tensor = src_graph.x.view(J, context, rotation_dim)

            losses = []
            for step in range(rollout):
                src_graph.x = context_tensor.reshape(J, context * rotation_dim).to(device)
                pred = model(src_graph, lastframe_graph)

                gt = tgt_graph.x[:, step * rotation_dim:(step + 1) * rotation_dim]

                loss = loss_fn(sixd_torch.to_matrix(pred.reshape(-1, 3, 2)),
                               sixd_torch.to_matrix(gt.reshape(-1, 3, 2)))
                losses.append(loss)

                context_tensor = (
                    torch.cat([context_tensor[:, 1:, :], pred.view(J, 1, rotation_dim).detach()], dim=1))
                lastframe_graph.x = pred.to(device)

            total_loss = sum(losses) / len(losses)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src_graph, lastframe_graph, tgt_graph in val_loader:
                src_graph = src_graph.to(device)
                lastframe_graph = lastframe_graph.to(device)
                tgt_graph = tgt_graph.to(device)

                J = src_graph.x.shape[0]
                context_tensor = src_graph.x.view(J, context, rotation_dim)

                losses = []
                for step in range(rollout):
                    src_graph.x = context_tensor.reshape(J, context * rotation_dim).to(device)
                    pred = model(src_graph, lastframe_graph)
                    gt = tgt_graph.x[:, step * rotation_dim:(step + 1) * rotation_dim]

                    loss = loss_fn(sixd_torch.to_matrix(pred.reshape(-1, 3, 2)),
                                   sixd_torch.to_matrix(gt.reshape(-1, 3, 2)))
                    losses.append(loss)

                    context_tensor = torch.cat([context_tensor[:, 1:, :], pred.view(J, 1, rotation_dim).detach()], dim=1)
                    lastframe_graph.x = pred.to(device)

                val_loss += (sum(losses) / len(losses)).item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model_" + time.strftime("%Y%m%d-%H%M%S") + ".pth"))
