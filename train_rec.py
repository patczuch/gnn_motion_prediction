import os
import torch
import time
import logging
import config

from torch_geometric.loader import DataLoader

from geodesicloss import GeodesicLoss
from motionpredictor import Model
from dataset import BVHMotionDataset
import pymotion.rotations.ortho6d_torch as sixd_torch


if __name__ == "__main__":
    start_time = time.strftime("%Y%m%d-%H%M%S")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    context = config.context_length

    os.makedirs(config.logs_dir, exist_ok=True)
    os.makedirs(config.checkpoints_dir, exist_ok=True)

    log_path = os.path.join(
        config.logs_dir, f"model_{start_time}_training.log"
    )
    logger = logging.getLogger("train_rec")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

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

    logger.info(
        f"Train size: {len(train_loader) * batch_size}, Validation size: {len(val_loader) * batch_size}"
    )

    rotation_dim = config.rotation_dim
    position_dim = config.position_dim
    feature_dim = rotation_dim + position_dim

    rollout = config.gen_frames
    pos_weight = config.pos_weight

    patience = config.early_stopping_patience
    min_delta = config.early_stopping_min_delta
    ckpt_interval = config.checkpoint_interval
    best_val = float('inf')
    epochs_no_improve = 0

    for epoch in range(config.epochs):
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
        logger.info(
            f"Epoch {epoch + 1:03d} | "
            f"Train Loss: {train_loss:.6f} "
            f"(rot: {train_rot_loss:.6f}, pos: {train_pos_loss:.6f}) | "
            f"Val Loss: {val_loss:.6f}"
        )

        if (epoch + 1) % ckpt_interval == 0:
            ckpt_name = f"model_{start_time}-{epoch + 1}.pth"
            torch.save(
                model.state_dict(),
                os.path.join(config.checkpoints_dir, ckpt_name),
            )
            logger.info(f"Saved checkpoint: {ckpt_name}")

        if val_loss + min_delta < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            best_name = f"model_{start_time}-best.pth"
            torch.save(
                model.state_dict(),
                os.path.join(config.checkpoints_dir, best_name),
            )
            logger.info(f"New best val loss {best_val:.6f}. Saved: {best_name}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch + 1}. Best Val Loss: {best_val:.6f}"
                )
                break

    final_name = f"model_{start_time}-final.pth"
    torch.save(
        model.state_dict(),
        os.path.join(config.checkpoints_dir, final_name),
    )
    logger.info(f"Training completed. Saved final model: {final_name}")