import math
import os
import random
import torch
import numpy as np
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
from plot_helpers import save_fk_3d_plots

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    checkpoint_path = "./checkpoints/model_20251228-114721-final.pth"
    dataset_path = "datasets/lafan1eval"
    out_dir = "./eval_results"
    os.makedirs(out_dir, exist_ok=True)

    context = config.context_length
    rollout = config.gen_frames
    num_samples = 10
    random.seed(10)

    rotation_dim = config.rotation_dim
    feature_dim = rotation_dim
    pos_weight = config.pos_weight

    dataset = BVHMotionDataset(dataset_path, context=context, step=context)

    model = Model().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    rot_loss_fn = GeodesicLoss(reduction='mean')
    pos_loss_fn = torch.nn.MSELoss(reduction='mean')

    print(f"Loaded dataset with {len(dataset)} samples")
    print(f"Testing model: {checkpoint_path}")
    print(f"Rollout steps: {rollout}")

    test_indices = random.sample(range(len(dataset)), num_samples)
    all_losses = []

    total_frames = context + rollout

    with torch.no_grad():
        for idx in test_indices:
            src_graph, lastframe_graph, tgt_graph = dataset[idx]

            src_graph = src_graph.to(device)
            lastframe_graph = lastframe_graph.to(device)
            tgt_graph = tgt_graph.to(device)

            J = src_graph.x.shape[0]
            context_frames = src_graph.x.view(J, context, feature_dim)

            losses = []

            filepath, start = dataset.samples[idx]
            feats = dataset.cache[filepath]  # (F, J, 9)

            bvh_rots = [
                feats[start + f, :, :rotation_dim].reshape(J, rotation_dim)
                for f in range(context)
            ]
            bvh_rots_gt = [
                feats[start + f, :, :rotation_dim].reshape(J, rotation_dim)
                for f in range(context)
            ]

            src_graph.x = context_frames.reshape(J, context * feature_dim).to(device)
            pred = model(src_graph, lastframe_graph)  # (J, rollout * 6)

            pred_seq = pred.view(J, rollout, feature_dim)  # (J, rollout, 6)
            gt_seq = tgt_graph.x[:, :rollout * feature_dim].view(J, rollout, feature_dim)

            for step in range(rollout):
                pred_step = pred_seq[:, step, :]  # (J, 6)

                pred_rot6 = pred_step

                pred_mat = sixd_torch.to_matrix(pred_rot6.view(-1, 3, 2))  # (J, 3, 3)
                pred_6d_norm = pred_mat[..., :3, :2].reshape(J, rotation_dim)

                bvh_rots.append(pred_6d_norm.cpu())

                gt_step = gt_seq[:, step, :]  # (J, 6)
                gt_rot6 = gt_step

                gt_mat = sixd_torch.to_matrix(gt_rot6.view(-1, 3, 2))  # (J, 3, 3)
                gt_6d_norm = gt_mat[..., :3, :2].reshape(J, rotation_dim)
                bvh_rots_gt.append(gt_6d_norm.cpu())

                rot_loss = rot_loss_fn(
                    pred_mat,
                    sixd_torch.to_matrix(gt_rot6.view(-1, 3, 2)),
                )

                pred_quat = quat_torch.from_matrix(
                    sixd_torch.to_matrix(pred_seq[:, : step + 1, :].reshape(-1, 3, 2))
                ).view(1, J, step + 1, 4)
                gt_quat = quat_torch.from_matrix(
                    sixd_torch.to_matrix(gt_seq[:, : step + 1, :].reshape(-1, 3, 2))
                ).view(1, J, step + 1, 4)

                pred_quat_tm = pred_quat.permute(0, 2, 1, 3)  # (B=1, T, J, 4)
                gt_quat_tm = gt_quat.permute(0, 2, 1, 3)  # (B=1, T, J, 4)

                parents_t = torch.tensor(dataset.parents, device=device).long()
                offsets_t = torch.tensor(dataset.offsets, device=device).to(pred.dtype)
                offsets_bt = offsets_t.view(1, 1, J, 3).expand(1, step + 1, J, 3)  # (B=1, T, J, 3)

                global_pos = torch.zeros((1, step + 1, 3), device=device, dtype=pred.dtype)

                pred_pos_tm, _ = skeleton_torch.fk(pred_quat_tm, global_pos, offsets_bt, parents_t)  # (1, T, J, 3)
                gt_pos_tm, _ = skeleton_torch.fk(gt_quat_tm, global_pos, offsets_bt, parents_t)  # (1, T, J, 3)

                #t_idx = step
                #plot_path = os.path.join(out_dir, "plots", f"sample_{os.path.basename(filepath).replace('.bvh','')}_{start}_step_{step:03d}.png")
                #save_fk_3d_plots(pred_pos_tm, gt_pos_tm, dataset.parents, plot_path, t_idx=t_idx)

                pos_loss = pos_weight * pos_loss_fn(pred_pos_tm, gt_pos_tm)

                loss = rot_loss + pos_loss
                losses.append(loss)
                print(
                    f"Sample {idx} frame {step} | "
                    f"rot_loss = {rot_loss:.6f} | pos_loss = {pos_loss:.6f} | "
                    f"total = {loss:.6f}"
                )

            avg_loss = sum(losses) / len(losses)
            all_losses.append(avg_loss)

            bvh_rots = torch.stack(bvh_rots)      # (total_frames, J, 6)
            bvh_rots_gt = torch.stack(bvh_rots_gt)  # (total_frames, J, 6)

            ortho6 = bvh_rots.view(-1, 3, 2).cpu().numpy()
            ortho6_gt = bvh_rots_gt.view(-1, 3, 2).cpu().numpy()
            bvh_rots = bvh_rots.to(torch.float32)
            bvh_rots_gt = bvh_rots_gt.to(torch.float32)
            T, Jb, D = bvh_rots.shape
            assert T == total_frames and Jb == J and D == rotation_dim

            quats = sixd.to_quat(ortho6)            # (T*J, 4)
            quats = quats.reshape(T, J, 4)          # (T, J, 4)
            quats_gt = sixd.to_quat(ortho6_gt)      # (T*J, 4)
            quats_gt = quats_gt.reshape(T, J, 4)    # (T, J, 4)

            eulers = quat.to_euler(
                quats,
                np.tile(dataset.rot_order, (T, 1, 1)),
            ) * 180 / math.pi
            eulers_gt = quat.to_euler(
                quats_gt,
                np.tile(dataset.rot_order, (T, 1, 1)),
            ) * 180 / math.pi

            bvh_gen = BVH()
            bvh_gen.data = {
                "names": dataset.names,
                "offsets": dataset.offsets,
                "end_sites": dataset.end_sites,
                "end_sites_parents": dataset.end_sites_parents,
                "parents": dataset.parents,
                "rot_order": dataset.rot_order,
                "positions": np.zeros((total_frames, len(dataset.names), 3)),
                "rotations": eulers,
                "frame_time": dataset.frame_time,
            }

            bvh_gt = BVH()
            bvh_gt.data = {
                "names": dataset.names,
                "offsets": dataset.offsets,
                "end_sites": dataset.end_sites,
                "end_sites_parents": dataset.end_sites_parents,
                "parents": dataset.parents,
                "rot_order": dataset.rot_order,
                "positions": np.zeros((total_frames, len(dataset.names), 3)),
                "rotations": eulers_gt,
                "frame_time": dataset.frame_time,
            }

            out_gen = os.path.join(
                out_dir,
                f"gen_{os.path.basename(filepath).replace('.bvh','')}_{start}.bvh",
            )
            bvh_gen.save(out_gen)

            out_gt = os.path.join(
                out_dir,
                f"gt_{os.path.basename(filepath).replace('.bvh','')}_{start}.bvh",
            )
            bvh_gt.save(out_gt)

            print(f"Exported sample {idx} →")
            print(f"  Generated:   {out_gen}")
            print(f"  GroundTruth: {out_gt}")
            print(f"  Average rollout loss = {avg_loss:.6f}")
            print("")

    print("-------------------------------------------------")
    print(
        f"Overall Mean loss over {num_samples} samples: "
        f"{sum(all_losses) / len(all_losses):.6f}"
    )
    print("Done!")