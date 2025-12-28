import os
import torch
import time
import logging
import config

from torch_geometric.loader import DataLoader
from torch.amp import autocast, GradScaler
from geodesicloss import GeodesicLoss
from motionpredictor import Model
from dataset import BVHMotionDataset
import pymotion.rotations.ortho6d_torch as sixd_torch
import pymotion.rotations.quat_torch as quat_torch
import pymotion.ops.skeleton_torch as skeleton_torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts



def compute_loss(pred, tgt_graph, src_graph, parents_t, offsets_t,
                 rot_loss_fn, pos_loss_fn, pos_weight, config):
    bone_length_dim = config.bone_length_dim
    rotation_dim = config.rotation_dim
    feature_dim = rotation_dim
    gen_frames = config.gen_frames

    gt_seq_flat = tgt_graph.x[:, bone_length_dim:bone_length_dim + gen_frames * feature_dim]

    BJ = pred.shape[0]
    B = int(src_graph.batch.max().item()) + 1 if hasattr(src_graph, 'batch') else 1
    J = BJ // B

    pred_seq = pred.view(B, J, gen_frames, feature_dim)
    gt_seq = gt_seq_flat.view(B, J, gen_frames, feature_dim)

    pred_rot6 = pred_seq.reshape(B * J * gen_frames, rotation_dim)
    gt_rot6 = gt_seq.reshape(B * J * gen_frames, rotation_dim)

    pred_R = sixd_torch.to_matrix(pred_rot6.view(-1, 3, 2)).view(B, J, gen_frames, 3, 3)
    gt_R = sixd_torch.to_matrix(gt_rot6.view(-1, 3, 2)).view(B, J, gen_frames, 3, 3)

    rot_loss = rot_loss_fn(pred_R.reshape(-1, 3, 3), gt_R.reshape(-1, 3, 3))

    pred_quat = quat_torch.from_matrix(pred_R.reshape(-1, 3, 3)).view(B, J, gen_frames, 4)
    gt_quat = quat_torch.from_matrix(gt_R.reshape(-1, 3, 3)).view(B, J, gen_frames, 4)

    pred_quat_tm = pred_quat.permute(0, 2, 1, 3)
    gt_quat_tm = gt_quat.permute(0, 2, 1, 3)

    offsets_bt = offsets_t.view(1, 1, J, 3).expand(B, gen_frames, J, 3)
    global_pos = torch.zeros((B, gen_frames, 3), device=pred.device, dtype=pred.dtype)

    pred_pos_tm, _ = skeleton_torch.fk(pred_quat_tm, global_pos, offsets_bt, parents_t)
    gt_pos_tm, _ = skeleton_torch.fk(gt_quat_tm, global_pos, offsets_bt, parents_t)

    pos_loss = pos_loss_fn(pred_pos_tm, gt_pos_tm)

    total_loss = rot_loss + pos_weight * pos_loss

    return total_loss, rot_loss, pos_loss * pos_weight

if __name__ == "__main__":
    start_time = time.strftime("%Y%m%d-%H%M%S")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    context = config.context_length

    os.makedirs(config.logs_dir, exist_ok=True)
    os.makedirs(config.checkpoints_dir, exist_ok=True)

    log_path = os.path.join(
        config.logs_dir, f"model_{start_time}_training.log"
    )

    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    #     stream=sys.stdout,
    # )

    logger = logging.getLogger("train")
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

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    model = Model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    rot_loss_fn = GeodesicLoss(reduction='mean')
    pos_loss_fn = torch.nn.MSELoss(reduction='mean')

    logger.info(
        f"Train size: {len(train_loader) * batch_size}, Validation size: {len(val_loader) * batch_size}"
    )

    rotation_dim = config.rotation_dim
    feature_dim = rotation_dim
    gen_frames = config.gen_frames

    pos_weight = config.pos_weight

    patience = config.early_stopping_patience
    min_delta = config.early_stopping_min_delta
    ckpt_interval = config.checkpoint_interval
    best_val = float('inf')
    epochs_no_improve = 0

    parents_t = torch.tensor(dataset.parents, device=device).long()
    offsets_t = torch.tensor(dataset.offsets, device=device).float()
    scaler = GradScaler('cuda')

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        train_rot_loss = 0.0
        train_pos_loss = 0.0

        for src_graph, lastframe_graph, tgt_graph in train_loader:
            src_graph = src_graph.to(device)
            lastframe_graph = lastframe_graph.to(device)
            tgt_graph = tgt_graph.to(device)

            optimizer.zero_grad()

            with autocast('cuda'):
                pred = model(src_graph, lastframe_graph)
                loss, rot_loss, pos_loss = compute_loss(
                    pred, tgt_graph, src_graph, parents_t, offsets_t,
                    rot_loss_fn, pos_loss_fn, pos_weight, config
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_rot_loss += rot_loss.item()
            train_pos_loss += pos_loss.item()

        train_loss /= len(train_loader)
        train_rot_loss /= len(train_loader)
        train_pos_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_rot_loss = 0.0
        val_pos_loss = 0.0

        with torch.no_grad():
            for src_graph, lastframe_graph, tgt_graph in val_loader:
                src_graph = src_graph.to(device)
                lastframe_graph = lastframe_graph.to(device)
                tgt_graph = tgt_graph.to(device)

                with autocast('cuda'):
                    pred = model(src_graph, lastframe_graph)
                    loss, rot_loss, pos_loss = compute_loss(
                        pred, tgt_graph, src_graph, parents_t, offsets_t,
                        rot_loss_fn, pos_loss_fn, pos_weight, config
                    )
                val_loss += loss.item()
                val_rot_loss += rot_loss.item()
                val_pos_loss += pos_loss.item()

        val_loss /= len(val_loader)
        val_rot_loss /= len(val_loader)
        val_pos_loss /= len(val_loader)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        logger.info(
            f"Epoch {epoch + 1:03d} | Train Loss: {train_loss:.6f} (rot: {train_rot_loss:.6f}, pos: {train_pos_loss:.6f}) | Val Loss: {val_loss:.6f} (rot: {val_rot_loss:.6f}, pos: {val_pos_loss:.6f}) | LR: {current_lr:.2e}"
        )

        if (epoch + 1) % ckpt_interval == 0:
            ckpt_name = f"model_{start_time}-{epoch + 1}.pth"
            torch.save(model.state_dict(), os.path.join(config.checkpoints_dir, ckpt_name))
            logger.info(f"Saved checkpoint: {ckpt_name}")

        if val_loss + min_delta < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            if epoch > ckpt_interval:
                best_name = f"model_{start_time}-best.pth"
                torch.save(model.state_dict(), os.path.join(config.checkpoints_dir, best_name))
                logger.info(f"New best val loss {best_val:.6f}. Saved: {best_name}")
            else:
                logger.info(f"New best val loss {best_val:.6f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch + 1}. Best Val Loss: {best_val:.6f}"
                )
                break

    final_name = f"model_{start_time}-final.pth"
    torch.save(model.state_dict(), os.path.join(config.checkpoints_dir, final_name))
    logger.info(f"Training completed. Saved final model: {final_name}")