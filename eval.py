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
from pymotion.io.bvh import BVH
import pymotion.rotations.quat as quat

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    checkpoint_path = "./checkpoints/model_20251130-173252.pth"
    dataset_path = "datasets/lafan1eval"
    out_dir = "./eval_results"
    os.makedirs(out_dir, exist_ok=True)

    context = 20
    rollout = 5
    num_samples = 5
    random.seed(10)

    rotation_dim = 6
    position_dim = 3
    feature_dim = rotation_dim + position_dim
    pos_weight = 0.1

    dataset = BVHMotionDataset(dataset_path, context=context, step=20)

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

            for step in range(rollout):
                src_graph.x = context_frames.reshape(J, context * feature_dim).to(device)
                pred = model(src_graph, lastframe_graph)  # (J, feature_dim)

                pred_rot6 = pred[:, :rotation_dim]
                pred_pos3 = pred[:, rotation_dim:rotation_dim + position_dim]

                pred_mat = sixd_torch.to_matrix(pred_rot6.view(-1, 3, 2))  # (J, 3, 3)
                pred_6d_norm = pred_mat[..., :3, :2].reshape(J, rotation_dim)

                bvh_rots.append(pred_6d_norm.cpu())

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

                loss = rot_loss + pos_weight * pos_loss
                losses.append(loss)
                print(
                    f"Sample {idx} frame {step} | "
                    f"rot_loss = {rot_loss:.6f} | pos_loss = {pos_loss:.6f} | "
                    f"total = {loss:.6f}"
                )

                pred_feat_next = torch.cat(
                    [pred_6d_norm, pred_pos3], dim=-1
                )  # (J, 9)

                context_frames = torch.cat(
                    [context_frames[:, 1:, :],
                     pred_feat_next.view(J, 1, feature_dim).detach()],
                    dim=1,
                )
                lastframe_graph.x = pred_feat_next.to(device)

            avg_loss = sum(losses) / len(losses)
            all_losses.append(avg_loss)

            bvh_rots = torch.stack(bvh_rots)  # (total_frames, J, 6)

            ortho6 = bvh_rots.view(-1, 3, 2).cpu().numpy()
            bvh_rots = bvh_rots.to(torch.float32)
            T, Jb, D = bvh_rots.shape
            assert T == total_frames and Jb == J and D == rotation_dim

            quats = sixd.to_quat(ortho6)        # (T*J, 4)
            quats = quats.reshape(T, J, 4)      # (T, J, 4)

            eulers = quat.to_euler(
                quats,
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

            out_gen = os.path.join(
                out_dir,
                f"gen_{os.path.basename(filepath).replace('.bvh','')}_{start}.bvh",
            )
            bvh_gen.save(out_gen)

            print(f"Exported sample {idx} →")
            print(f"  Generated:   {out_gen}")
            print(f"  Average rollout loss = {avg_loss:.6f}")
            print("")

    print("-------------------------------------------------")
    print(
        f"Overall Mean loss over {num_samples} samples: "
        f"{sum(all_losses) / len(all_losses):.6f}"
    )
    print("Done!")