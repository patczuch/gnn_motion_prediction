import os
import torch
import time
import logging
import config
import random

from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader
from torch.amp import autocast, GradScaler
from geodesicloss import GeodesicLoss
from motionpredictor import Model
from dataset import BVHMotionDataset, split_train_val
import pymotion.rotations.ortho6d_torch as sixd_torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy


class GroupedSkeletonBatchSampler(Sampler):
    def __init__(self, skeleton_ids, batch_size, shuffle=True, drop_last=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        from collections import defaultdict
        self.groups = defaultdict(list)
        for idx, sid in enumerate(skeleton_ids):
            self.groups[sid].append(idx)

    def __iter__(self):
        all_batches = []
        for sid, indices in self.groups.items():
            if self.shuffle:
                indices = indices.copy()
                random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)
        if self.shuffle:
            random.shuffle(all_batches)
        yield from all_batches

    def __len__(self):
        total = 0
        for sid, indices in self.groups.items():
            n = len(indices)
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total


def positions_from_global_rotmats(global_rotmats, global_pos, offsets, parents):
    B, T, J = global_rotmats.shape[:3]
    positions = torch.zeros(B, T, J, 3, device=global_rotmats.device, dtype=global_rotmats.dtype)
    positions[:, :, 0, :] = global_pos
    for j in range(1, J):
        p = parents[j].item()
        rotated_offset = torch.einsum('btij,bj->bti', global_rotmats[:, :, p, :, :], offsets[:, j, :])
        positions[:, :, j, :] = positions[:, :, p, :] + rotated_offset
    return positions


def compute_loss(pred_rot, pred_root_pos, tgt_graph, src_graph,
                 rot_loss_fn, pos_loss_fn, rot_weight, pos_weight, smooth_weight, root_pos_weight, transition_weight, config):
    rotation_dim = config.rotation_dim
    feature_dim = rotation_dim
    gen_frames = config.gen_frames
    context_length = config.context_length

    gt_seq_flat = tgt_graph.x[:, 0:gen_frames * feature_dim]

    BJ = pred_rot.shape[0]
    B = int(src_graph.batch.max().item()) + 1 if hasattr(src_graph, 'batch') else 1
    J = BJ // B

    pred_seq = pred_rot.view(B, J, gen_frames, feature_dim)
    gt_seq = gt_seq_flat.view(B, J, gen_frames, feature_dim)

    pred_rot6 = pred_seq.reshape(B * J * gen_frames, rotation_dim)
    gt_rot6 = gt_seq.reshape(B * J * gen_frames, rotation_dim)

    pred_R = sixd_torch.to_matrix(pred_rot6.view(-1, 3, 2)).view(B, J, gen_frames, 3, 3)
    gt_R = sixd_torch.to_matrix(gt_rot6.view(-1, 3, 2)).view(B, J, gen_frames, 3, 3)

    rot_loss = rot_loss_fn(pred_R.reshape(-1, 3, 3), gt_R.reshape(-1, 3, 3))

    parents_t = src_graph.parents[:J]

    offsets_b = src_graph.offsets.view(B, J, 3)

    pred_R_tm = pred_R.permute(0, 2, 1, 3, 4)  # (B, gen_frames, J, 3, 3)
    gt_R_tm = gt_R.permute(0, 2, 1, 3, 4)

    global_pos = torch.zeros((B, gen_frames, 3), device=pred_rot.device, dtype=pred_rot.dtype)

    pred_pos = positions_from_global_rotmats(pred_R_tm, global_pos, offsets_b, parents_t)
    gt_pos = positions_from_global_rotmats(gt_R_tm, global_pos, offsets_b, parents_t)

    pos_loss = pos_loss_fn(pred_pos, gt_pos)

    # Smoothness loss
    src_seq_flat = src_graph.x[:, 0:context_length * feature_dim]
    src_seq = src_seq_flat.view(B, J, context_length, feature_dim)
    last_ctx_rot6 = src_seq[:, :, -1, :]

    last_ctx_R = sixd_torch.to_matrix(last_ctx_rot6.reshape(-1, 3, 2)).view(B, J, 1, 3, 3)
    last_ctx_R_tm = last_ctx_R.permute(0, 2, 1, 3, 4)  # (B, 1, J, 3, 3)

    global_pos_single = torch.zeros((B, 1, 3), device=pred_rot.device, dtype=pred_rot.dtype)
    last_ctx_pos = positions_from_global_rotmats(last_ctx_R_tm, global_pos_single, offsets_b, parents_t)

    pred_pos_with_ctx = torch.cat([last_ctx_pos, pred_pos], dim=1)
    gt_pos_with_ctx = torch.cat([last_ctx_pos, gt_pos], dim=1)

    pred_vel = pred_pos_with_ctx[:, 1:, :, :] - pred_pos_with_ctx[:, :-1, :, :]
    gt_vel = gt_pos_with_ctx[:, 1:, :, :] - gt_pos_with_ctx[:, :-1, :, :]

    transition_loss = pos_loss_fn(pred_vel[:, 0:1, :, :], gt_vel[:, 0:1, :, :])
    inner_smooth_loss = pos_loss_fn(pred_vel[:, 1:, :, :], gt_vel[:, 1:, :, :])
    smooth_loss = transition_weight * transition_loss + inner_smooth_loss

    # Root position loss
    gt_root_pos = tgt_graph.root_pos.view(B, gen_frames * 3)
    root_pos_loss = pos_loss_fn(pred_root_pos, gt_root_pos)

    total_loss = (rot_weight * rot_loss
                  + pos_weight * pos_loss
                  + smooth_weight * smooth_loss
                  + root_pos_weight * root_pos_loss)

    return total_loss, rot_weight * rot_loss, pos_loss * pos_weight, smooth_loss * smooth_weight, root_pos_loss * root_pos_weight

if __name__ == "__main__":
    start_time = time.strftime("%Y%m%d-%H%M%S")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    context = config.context_length

    torch.set_printoptions(profile="full")
    numpy.set_printoptions(threshold=10_000)

    os.makedirs(config.logs_dir, exist_ok=True)
    os.makedirs(config.checkpoints_dir, exist_ok=True)

    log_path = os.path.join(
        config.logs_dir, f"model_{start_time}_training.log"
    )

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

    dataset = BVHMotionDataset(config.train_data_paths,
                               context=context, step=config.step_size)

    val_ratio = 0.2
    # Seeded so that resuming from a checkpoint keeps the same split; otherwise
    # the previous run's validation samples become training samples.
    train_indices, val_indices, dropped_boundary = split_train_val(
        dataset, val_ratio=val_ratio, seed=config.split_seed
    )

    batch_size = 512

    train_skel_ids = [dataset.sample_skel[i] for i in train_indices]
    val_skel_ids = [dataset.sample_skel[i] for i in val_indices]

    train_sampler = GroupedSkeletonBatchSampler(
        skeleton_ids=train_skel_ids, batch_size=batch_size, shuffle=True
    )
    val_sampler = GroupedSkeletonBatchSampler(
        skeleton_ids=val_skel_ids, batch_size=batch_size, shuffle=False
    )

    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_set, batch_sampler=train_sampler,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_sampler=val_sampler,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    model = Model().to(device)

    logger.info(f"Using config: {config.config_file}")

    if config.checkpoint_path is not None and len(config.checkpoint_path) > 0:
        logger.info(f"Resuming from checkpoint: {config.checkpoint_path}")
        checkpoint = torch.load(config.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                  threshold=config.early_stopping_min_delta, threshold_mode='abs')

    rot_loss_fn = GeodesicLoss(reduction='mean')
    pos_loss_fn = torch.nn.MSELoss(reduction='mean')

    logger.info(
        f"Train size: {len(train_indices)}, Validation size: {len(val_indices)}, "
        f"Dropped at split boundaries: {dropped_boundary}, "
        f"Unique skeletons: {len(dataset._skel_key_to_id)}"
    )

    # Record every hyperparameter that shaped this run, including the ones that
    # live only in this file, so a log can be compared against later runs.
    logger.info(
        "Hyperparameters | " + ", ".join(
            f"{k}={getattr(config, k)}" for k in (
                "step_size", "context_length", "gen_frames", "hid_lyrs", "head_num",
                "dropout", "facing_normalization", "skeleton_geometry", "split_seed",
                "rot_weight", "pos_weight", "smooth_weight", "transition_weight",
                "root_pos_weight", "grad_clip_norm",
                "early_stopping_patience", "early_stopping_min_delta", "epochs",
            )
        )
        + f", batch_size={batch_size}, lr={optimizer.param_groups[0]['lr']}"
        + f", val_ratio={val_ratio}, params={sum(p.numel() for p in model.parameters())}"
    )

    rotation_dim = config.rotation_dim
    feature_dim = rotation_dim
    gen_frames = config.gen_frames

    rot_weight = config.rot_weight
    pos_weight = config.pos_weight
    smooth_weight = config.smooth_weight
    root_pos_weight = config.root_pos_weight
    transition_weight = config.transition_weight

    patience = config.early_stopping_patience
    min_delta = config.early_stopping_min_delta
    ckpt_interval = config.checkpoint_interval
    # Interval checkpoints exist only for disaster recovery; keep_last_checkpoints
    # caps how many are retained (0 = keep all). -best/-final are never pruned.
    keep_last = config.keep_last_checkpoints
    interval_ckpts = []
    best_val = float('inf')          # for saving the best checkpoint (strict)
    best_val_patience = float('inf')  # for early stopping (needs min_delta)
    epochs_no_improve = 0

    scaler = GradScaler('cuda')

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        train_rot_loss = 0.0
        train_pos_loss = 0.0
        train_root_pos_loss = 0.0
        steps = 0

        for src_graph, tgt_graph in train_loader:
            src_graph = src_graph.to(device)
            tgt_graph = tgt_graph.to(device)

            optimizer.zero_grad()

            with autocast('cuda'):
                pred_rot, pred_root_pos = model(src_graph)

            with autocast('cuda', enabled=False):
                loss, rot_loss, pos_loss, smooth_loss, root_pos_loss = compute_loss(
                    pred_rot.float(), pred_root_pos.float(), tgt_graph, src_graph,
                    rot_loss_fn, pos_loss_fn, rot_weight, pos_weight, smooth_weight, root_pos_weight, transition_weight, config
                )

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss detected at epoch {epoch + 1}. Skipping batch to prevent weight corruption.")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_rot_loss += rot_loss.item()
            train_pos_loss += pos_loss.item()
            train_root_pos_loss += root_pos_loss.item()
            steps += 1

        if steps > 0:
            train_loss /= steps
            train_rot_loss /= steps
            train_pos_loss /= steps
            train_root_pos_loss /= steps
        else:
            logger.warning(f"Epoch {epoch + 1}: All batches resulted in NaN/Inf loss!")

        model.eval()
        val_loss = 0.0
        val_rot_loss = 0.0
        val_pos_loss = 0.0
        val_smooth_loss = 0.0
        val_root_pos_loss = 0.0

        with torch.no_grad():
            for src_graph, tgt_graph in val_loader:
                src_graph = src_graph.to(device)
                tgt_graph = tgt_graph.to(device)

                with autocast('cuda'):
                    pred_rot, pred_root_pos = model(src_graph)

                with autocast('cuda', enabled=False):
                    loss, rot_loss, pos_loss, smooth_loss, root_pos_loss = compute_loss(
                        pred_rot.float(), pred_root_pos.float(), tgt_graph, src_graph,
                        rot_loss_fn, pos_loss_fn, rot_weight, pos_weight, smooth_weight, root_pos_weight, transition_weight, config
                    )
                val_loss += loss.item()
                val_rot_loss += rot_loss.item()
                val_pos_loss += pos_loss.item()
                val_smooth_loss += smooth_loss.item()
                val_root_pos_loss += root_pos_loss.item()

        val_loss /= len(val_loader)
        val_rot_loss /= len(val_loader)
        val_pos_loss /= len(val_loader)
        val_smooth_loss /= len(val_loader)
        val_root_pos_loss /= len(val_loader)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        logger.info(
            f"Epoch {epoch + 1:03d} | Train Loss: {train_loss:.6f} (rot: {train_rot_loss:.6f}, pos: {train_pos_loss:.6f}, root: {train_root_pos_loss:.6f}) | "
            f"Val Loss: {val_loss:.6f} (rot: {val_rot_loss:.6f}, pos: {val_pos_loss:.6f}, smooth: {val_smooth_loss:.6f}, root: {val_root_pos_loss:.6f}) | LR: {current_lr:.2e}"
        )

        if (epoch + 1) % ckpt_interval == 0:
            ckpt_name = f"model_{start_time}-{epoch + 1}.pth"
            ckpt_path = os.path.join(config.checkpoints_dir, ckpt_name)
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_name}")

            # Prune only this run's interval checkpoints, oldest first, and only
            # after the new one is on disk so a crash never leaves us with none.
            interval_ckpts.append(ckpt_path)
            while keep_last > 0 and len(interval_ckpts) > keep_last:
                stale = interval_ckpts.pop(0)
                try:
                    os.remove(stale)
                    logger.info(f"Pruned old checkpoint: {os.path.basename(stale)}")
                except OSError as e:
                    logger.warning(f"Could not prune {os.path.basename(stale)}: {e}")

        if val_loss < best_val:
            best_val = val_loss
            best_name = f"model_{start_time}-best.pth"
            torch.save(model.state_dict(), os.path.join(config.checkpoints_dir, best_name))
            logger.info(f"New best val loss {best_val:.6f}. Saved: {best_name}")

        if val_loss + min_delta < best_val_patience:
            best_val_patience = val_loss
            epochs_no_improve = 0
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