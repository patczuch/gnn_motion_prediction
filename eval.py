import math
import os
import random
import torch
import numpy as np
from torch_geometric.data import Data
from motionpredictor import Model
from dataset import BVHMotionDataset
from geodesicloss import GeodesicLoss
import pymotion.rotations.ortho6d as sixd
import pymotion.rotations.ortho6d_torch as sixd_torch
import pymotion.rotations.quat_torch as quat_torch
import pymotion.ops.skeleton_torch as skeleton_torch
from pymotion.io.bvh import BVH
import pymotion.rotations.quat as quat
import config

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    checkpoint_path = "./checkpoints/model_20260329-163748-best.pth"
    dataset_paths = config.eval_data_paths

    out_dir = "./eval_results"
    os.makedirs(out_dir, exist_ok=True)

    context = config.context_length
    gen_frames = config.gen_frames
    rollout = gen_frames
    num_samples = 30
    random.seed(10)

    rotation_dim = config.rotation_dim
    pos_weight = config.pos_weight

    dataset = BVHMotionDataset(dataset_paths, context=context, step=config.step_size)

    model = Model().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    rot_loss_fn = GeodesicLoss(reduction='mean')
    pos_loss_fn = torch.nn.SmoothL1Loss(reduction='mean')

    print(f"Loaded dataset with {len(dataset)} samples")
    print(f"Testing model: {checkpoint_path}")

    test_indices = random.sample(range(len(dataset)), num_samples)
    all_losses = []

    total_frames = context + rollout

    with torch.no_grad():
        for idx in test_indices:
            filepath, start = dataset.samples[idx]
            feats = dataset.cache[filepath]  # (F, J, C)
            root_positions = dataset.root_pos_cache[filepath]  # (F, 3)
            skeleton = dataset.get_skeleton(filepath)
            J = feats.shape[1]

            if start + total_frames > feats.shape[0]:
                print(f"Skipping sample {idx}: not enough frames (need {total_frames}, have {feats.shape[0] - start})")
                continue

            all_gt = feats[start : start + total_frames]  # (total_frames, J, C)
            all_gt_rot = all_gt[:, :, :rotation_dim]

            # --- Root positions (delta from first context frame) ---
            root_pos_window = root_positions[start:start + total_frames]  # (total_frames, 3)
            root_pos_origin = root_pos_window[0:1]  # (1, 3)
            root_pos_window_delta = root_pos_window - root_pos_origin

            root_pos_context = root_pos_window_delta[:context].reshape(context * 3)
            root_pos_gt_delta = root_pos_window_delta[context:context + gen_frames]  # (gen_frames, 3)

            all_gt_rot_orig = all_gt[:, :, :rotation_dim]

            bvh_rots = [
                all_gt_rot[f].clone() for f in range(context)
            ]
            bvh_rots_gt = [
                all_gt_rot_orig[f].clone() for f in range(total_frames)
            ]

            generated_frames = []  # list of (J, rotation_dim) tensors

            losses = []

            n_generated = len(generated_frames)

            if n_generated == 0:
                context_frames = all_gt_rot[:context]
            else:
                all_available = torch.cat(
                    [all_gt_rot[:context]] + [g.unsqueeze(0) for g in generated_frames],
                    dim=0
                )
                context_frames = all_available[-context:]

            # Prepare graph input
            x_input = context_frames.permute(1, 0, 2).reshape(J, -1).to(device)
            edge_index = skeleton["edges"].to(device)
            batch = torch.zeros(J, dtype=torch.long, device=device)
            src_graph = Data(
                x=x_input, edge_index=edge_index, batch=batch,
                root_pos=root_pos_context.to(device),
            )

            pred_rot, pred_root_pos = model(src_graph)
            pred_seq = pred_rot.view(J, gen_frames, rotation_dim)

            # Predicted root pos delta: (gen_frames, 3)
            pred_root_pos_delta = pred_root_pos.view(gen_frames, 3)

            gt_start = context
            gt_end = gt_start + gen_frames
            gt_chunk = all_gt_rot[gt_start:gt_end]
            gt_chunk_dev = gt_chunk.to(device)
            gt_seq = gt_chunk_dev.permute(1, 0, 2)

            for step in range(gen_frames):
                global_step = context + step
                pred_step = pred_seq[:, step, :]
                pred_mat = sixd_torch.to_matrix(pred_step.view(-1, 3, 2))
                pred_6d_norm = pred_mat[..., :3, :2].reshape(J, rotation_dim)
                bvh_rots.append(pred_6d_norm.cpu())
                generated_frames.append(pred_6d_norm.cpu())

                gt_step = gt_seq[:, step, :]
                gt_mat = sixd_torch.to_matrix(gt_step.view(-1, 3, 2))

                rot_loss = rot_loss_fn(pred_mat, gt_mat)

                n_so_far = len(generated_frames)
                all_gen_tensor = torch.stack(
                    [g.to(device) for g in generated_frames], dim=1
                )
                all_gt_target = all_gt_rot[context:context + n_so_far].to(device).permute(1, 0, 2)

                pred_quat = quat_torch.from_matrix(
                    sixd_torch.to_matrix(all_gen_tensor.reshape(-1, 3, 2))
                ).view(1, J, n_so_far, 4)
                gt_quat = quat_torch.from_matrix(
                    sixd_torch.to_matrix(all_gt_target.reshape(-1, 3, 2))
                ).view(1, J, n_so_far, 4)

                pred_quat_tm = pred_quat.permute(0, 2, 1, 3)
                gt_quat_tm = gt_quat.permute(0, 2, 1, 3)

                parents_t = torch.tensor(skeleton["parents"], device=device).long()
                offsets_t = torch.tensor(skeleton["offsets"], device=device).to(pred_rot.dtype)
                offsets_bt = offsets_t.view(1, 1, J, 3).expand(1, n_so_far, J, 3)

                # Use predicted/gt root pos deltas for FK
                pred_global_pos = pred_root_pos_delta[:n_so_far].unsqueeze(0)  # (1, n_so_far, 3)
                gt_global_pos = root_pos_gt_delta[:n_so_far].to(device).unsqueeze(0)  # (1, n_so_far, 3)

                pred_pos_tm, _ = skeleton_torch.fk(pred_quat_tm, pred_global_pos, offsets_bt, parents_t)
                gt_pos_tm, _ = skeleton_torch.fk(gt_quat_tm, gt_global_pos, offsets_bt, parents_t)

                pos_loss = pos_weight * pos_loss_fn(pred_pos_tm, gt_pos_tm)
                loss = rot_loss + pos_loss
                losses.append(loss)
                print(
                    f"Sample {idx} frame {global_step} | "
                    f"rot_loss = {rot_loss:.6f} | pos_loss = {pos_loss:.6f} | "
                    f"total = {loss:.6f}"
                )

            avg_loss = sum(losses) / len(losses)
            all_losses.append(avg_loss)

            bvh_rots = torch.stack(bvh_rots)
            bvh_rots_gt = torch.stack(bvh_rots_gt)

            ortho6 = bvh_rots.view(-1, 3, 2).cpu().numpy()
            ortho6_gt = bvh_rots_gt.view(-1, 3, 2).cpu().numpy()
            bvh_rots = bvh_rots.to(torch.float32)
            bvh_rots_gt = bvh_rots_gt.to(torch.float32)

            T_total, Jb, D = bvh_rots.shape
            assert T_total == total_frames and Jb == J and D == rotation_dim

            quats = sixd.to_quat(ortho6).reshape(T_total, J, 4)
            quats_gt = sixd.to_quat(ortho6_gt).reshape(T_total, J, 4)

            eulers = quat.to_euler(
                quats,
                np.tile(skeleton["rot_order"], (T_total, 1, 1)),
            ) * 180 / math.pi
            eulers_gt = quat.to_euler(
                quats_gt,
                np.tile(skeleton["rot_order"], (T_total, 1, 1)),
            ) * 180 / math.pi

            # Build root positions for BVH export (absolute)
            # Context: GT absolute, Generated: predicted delta + origin
            ctx_root_abs = root_pos_window[:context].numpy()  # (context, 3) — absolute
            pred_root_abs = (pred_root_pos_delta.cpu() + root_pos_origin).numpy()  # (gen_frames, 3)
            gt_root_abs = root_pos_window.numpy()  # (total_frames, 3) — absolute

            bvh_positions_gen = np.zeros((total_frames, len(skeleton["names"]), 3))
            bvh_positions_gen[:context, 0, :] = ctx_root_abs
            bvh_positions_gen[context:, 0, :] = pred_root_abs

            bvh_positions_gt = np.zeros((total_frames, len(skeleton["names"]), 3))
            bvh_positions_gt[:, 0, :] = gt_root_abs

            bvh_gen = BVH()
            bvh_gen.data = {
                "names": skeleton["names"],
                "offsets": skeleton["offsets"],
                "end_sites": skeleton["end_sites"],
                "end_sites_parents": skeleton["end_sites_parents"],
                "parents": skeleton["parents"],
                "rot_order": skeleton["rot_order"],
                "positions": bvh_positions_gen,
                "rotations": eulers,
                "frame_time": skeleton["frame_time"],
            }

            bvh_gt = BVH()
            bvh_gt.data = {
                "names": skeleton["names"],
                "offsets": skeleton["offsets"],
                "end_sites": skeleton["end_sites"],
                "end_sites_parents": skeleton["end_sites_parents"],
                "parents": skeleton["parents"],
                "rot_order": skeleton["rot_order"],
                "positions": bvh_positions_gt,
                "rotations": eulers_gt,
                "frame_time": skeleton["frame_time"],
            }

            out_gen = os.path.join(
                out_dir,
                f"gen_{os.path.basename(filepath).replace('.bvh', '')}_{start}.bvh",
            )
            bvh_gen.save(out_gen)

            out_gt = os.path.join(
                out_dir,
                f"gt_{os.path.basename(filepath).replace('.bvh', '')}_{start}.bvh",
            )
            bvh_gt.save(out_gt)

            print(f"Exported sample {idx} →")
            print(f"  Generated:   {out_gen}")
            print(f"  GroundTruth: {out_gt}")
            print(f"  Average rollout loss = {avg_loss:.6f}")
            print("")

    print("-------------------------------------------------")
    print(
        f"Overall Mean loss over {len(all_losses)} samples: "
        f"{sum(all_losses) / len(all_losses):.6f}"
    )
    print("Done!")