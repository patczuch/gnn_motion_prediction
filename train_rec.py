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

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    model = Model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    rot_loss_fn = GeodesicLoss(reduction="mean")
    pos_loss_fn = torch.nn.MSELoss(reduction="mean")

    print(
        f"Train size: {len(train_loader) * batch_size}, "
        f"Validation size: {len(val_loader) * batch_size}"
    )

    rotation_dim = 6
    position_dim = 3
    feature_dim = rotation_dim + position_dim

    rollout = 5
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

            J = src_graph.x.shape[0]
            context_frames = src_graph.x.view(J, context, feature_dim)

            step_rot_losses = []
            step_pos_losses = []

            for step in range(rollout):
                src_graph.x = context_frames.reshape(J, context * feature_dim).to(device)
                pred = model(src_graph, lastframe_graph)  # (J, feature_dim)

                pred_rot6 = pred[:, :rotation_dim]
                pred_pos3 = pred[:, rotation_dim:rotation_dim + position_dim]

                pred_mat = sixd_torch.to_matrix(pred_rot6.view(-1, 3, 2))  # (J, 3, 3)
                pred_6d_norm = pred_mat[..., :3, :2].reshape(J, rotation_dim)

                gt_step = tgt_graph.x[
                    :, step * feature_dim : (step + 1) * feature_dim
                ]  # (J, 9)

                gt_rot6 = gt_step[:, :rotation_dim]
                gt_pos3 = gt_step[:, rotation_dim:rotation_dim + position_dim]

                rot_loss = rot_loss_fn(
                    pred_mat,
                    sixd_torch.to_matrix(gt_rot6.view(-1, 3, 2)),
                )
                pos_loss = pos_loss_fn(pred_pos3, gt_pos3)

                step_rot_losses.append(rot_loss)
                step_pos_losses.append(pos_loss)

                pred_feat_next = torch.cat([pred_6d_norm, pred_pos3], dim=-1)  # (J, 9)

                context_frames = torch.cat(
                    [
                        context_frames[:, 1:, :],
                        pred_feat_next.view(J, 1, feature_dim).detach(),
                    ],
                    dim=1,
                )
                lastframe_graph.x = pred_feat_next.to(device)

            avg_rot_loss = sum(step_rot_losses) / len(step_rot_losses)
            avg_pos_loss = sum(step_pos_losses) / len(step_pos_losses)
            loss = avg_rot_loss + pos_weight * avg_pos_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_rot_loss += avg_rot_loss.detach()
            train_pos_loss += (pos_weight * avg_pos_loss).detach()

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

                J = src_graph.x.shape[0]
                context_frames = src_graph.x.view(J, context, feature_dim)

                step_rot_losses = []
                step_pos_losses = []

                for step in range(rollout):
                    src_graph.x = context_frames.reshape(J, context * feature_dim).to(device)
                    pred = model(src_graph, lastframe_graph)  # (J, feature_dim)

                    pred_rot6 = pred[:, :rotation_dim]
                    pred_pos3 = pred[:, rotation_dim:rotation_dim + position_dim]

                    pred_mat = sixd_torch.to_matrix(pred_rot6.view(-1, 3, 2))
                    pred_6d_norm = pred_mat[..., :3, :2].reshape(J, rotation_dim)

                    gt_step = tgt_graph.x[
                        :, step * feature_dim : (step + 1) * feature_dim
                    ]
                    gt_rot6 = gt_step[:, :rotation_dim]
                    gt_pos3 = gt_step[:, rotation_dim:rotation_dim + position_dim]

                    rot_loss = rot_loss_fn(
                        pred_mat,
                        sixd_torch.to_matrix(gt_rot6.view(-1, 3, 2)),
                    )
                    pos_loss = pos_loss_fn(pred_pos3, gt_pos3)

                    step_rot_losses.append(rot_loss)
                    step_pos_losses.append(pos_loss)

                    pred_feat_next = torch.cat(
                        [pred_6d_norm, pred_pos3], dim=-1
                    )  # (J, 9)

                    context_frames = torch.cat(
                        [
                            context_frames[:, 1:, :],
                            pred_feat_next.view(J, 1, feature_dim),
                        ],
                        dim=1,
                    )
                    lastframe_graph.x = pred_feat_next.to(device)

                avg_rot_loss = sum(step_rot_losses) / len(step_rot_losses)
                avg_pos_loss = sum(step_pos_losses) / len(step_pos_losses)
                loss = avg_rot_loss + pos_weight * avg_pos_loss

                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train Loss: {train_loss:.6f} "
            f"(rot: {train_rot_loss:.6f}, pos: {train_pos_loss:.6f}) | "
            f"Val Loss: {val_loss:.6f}"
        )

    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(save_dir, "model_" + time.strftime("%Y%m%d-%H%M%S") + ".pth"),
    )