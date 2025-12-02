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

    dataset = BVHMotionDataset("./datasets/lafan1train", context=context, step=context)

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
    rot_loss_fn = GeodesicLoss(reduction='mean')
    pos_loss_fn = torch.nn.MSELoss(reduction='mean')

    print(f"Train size: {len(train_loader) * batch_size}, Validation size: {len(val_loader) * batch_size}")

    rotation_dim = 6
    position_dim = 3
    feature_dim = rotation_dim + position_dim
    gen_frames = 5

    pos_weight = 0.03

    for epoch in range(200):
        model.train()
        train_loss = 0.0
        train_rot_loss = 0.0
        train_pos_loss = 0.0

        for src_graph, lastframe_graph, tgt_graph in train_loader:
            src_graph = src_graph.to(device)
            lastframe_graph = lastframe_graph.to(device)
            tgt_graph = tgt_graph.to(device)

            pred = model(src_graph, lastframe_graph)

            gt_seq_flat = tgt_graph.x[:, :gen_frames * feature_dim]  # (J, gen_frames * feature_dim)

            pred_seq = pred.view(-1, gen_frames, feature_dim)
            gt_seq = gt_seq_flat.view(-1, gen_frames, feature_dim)

            pred_rot6 = pred_seq[..., :rotation_dim].reshape(-1, rotation_dim)
            gt_rot6 = gt_seq[..., :rotation_dim].reshape(-1, rotation_dim)

            pred_pos3 = pred_seq[..., rotation_dim:rotation_dim + position_dim].reshape(-1, position_dim)
            gt_pos3 = gt_seq[..., rotation_dim:rotation_dim + position_dim].reshape(-1, position_dim)

            rot_loss = rot_loss_fn(
                sixd_torch.to_matrix(pred_rot6.reshape(-1, 3, 2)),
                sixd_torch.to_matrix(gt_rot6.reshape(-1, 3, 2))
            )

            train_rot_loss += rot_loss

            pos_loss = pos_loss_fn(pred_pos3, gt_pos3)

            train_pos_loss += pos_weight * pos_loss

            loss = rot_loss + pos_weight * pos_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_rot_loss /= len(train_loader)
        train_pos_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src_graph, lastframe_graph, tgt_graph in val_loader:
                src_graph = src_graph.to(device)
                lastframe_graph = lastframe_graph.to(device)
                tgt_graph = tgt_graph.to(device)

                pred = model(src_graph, lastframe_graph)

                gt_seq_flat = tgt_graph.x[:, :gen_frames * feature_dim]

                pred_seq = pred.view(-1, gen_frames, feature_dim)
                gt_seq = gt_seq_flat.view(-1, gen_frames, feature_dim)

                pred_rot6 = pred_seq[..., :rotation_dim].reshape(-1, rotation_dim)
                gt_rot6 = gt_seq[..., :rotation_dim].reshape(-1, rotation_dim)

                pred_pos3 = pred_seq[..., rotation_dim:rotation_dim + position_dim].reshape(-1, position_dim)
                gt_pos3 = gt_seq[..., rotation_dim:rotation_dim + position_dim].reshape(-1, position_dim)

                rot_loss = rot_loss_fn(
                    sixd_torch.to_matrix(pred_rot6.reshape(-1, 3, 2)),
                    sixd_torch.to_matrix(gt_rot6.reshape(-1, 3, 2))
                )
                pos_loss = pos_loss_fn(pred_pos3, gt_pos3)

                loss = rot_loss + pos_weight * pos_loss
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1:03d} | Train Loss: {train_loss:.6f} (rot: {train_rot_loss:.6f}, pos: {train_pos_loss:.6f}) | Val Loss: {val_loss:.6f}")

    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model_" + time.strftime("%Y%m%d-%H%M%S") + ".pth"))