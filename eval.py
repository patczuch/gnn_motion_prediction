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

    checkpoint_path = "./checkpoints/model_20251130-144952.pth"
    dataset_path = "datasets/lafan1eval"
    out_dir = "./eval_results"
    os.makedirs(out_dir, exist_ok=True)

    context = 20
    rollout = 5
    rotation_dim = 6
    num_samples = 5
    random.seed(10)

    dataset = BVHMotionDataset(dataset_path, context=context, step=20)

    model = Model().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    loss_fn = GeodesicLoss(reduction='mean')

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
            context_frames = src_graph.x.view(J, context, rotation_dim)

            losses = []

            filepath, start = dataset.samples[idx]
            rot = dataset.cache[filepath]

            bvh_rots = [rot[start + f].reshape(J, 6) for f in range(context)]

            for step in range(rollout):
                src_graph.x = context_frames.reshape(J, context * rotation_dim).to(device)
                pred = model(src_graph, lastframe_graph)

                pred_mat = sixd_torch.to_matrix(pred.view(-1, 3, 2))
                pred_6d = pred_mat[..., :3, :2].reshape(J, 6)

                bvh_rots.append(pred_6d.cpu())

                gt = tgt_graph.x[:, step * rotation_dim:(step + 1) * rotation_dim]

                loss = loss_fn(
                    pred_mat,
                    sixd_torch.to_matrix(gt.view(-1, 3, 2))
                )
                losses.append(loss)
                print(f"Sample {idx} frame {step} | Loss = {loss:.6f}")

                context_frames = torch.cat(
                    [context_frames[:, 1:, :],
                     pred_6d.view(J, 1, rotation_dim).detach()],
                    dim=1
                )
                lastframe_graph.x = pred_6d.to(device)

            avg_loss = sum(losses) / len(losses)
            all_losses.append(avg_loss)

            bvh_rots = torch.stack(bvh_rots)

            ortho6 = bvh_rots.view(-1, 3, 2).cpu().numpy()

            bvh_rots = bvh_rots.to(torch.float32)
            T, J, D = bvh_rots.shape

            quats = sixd.to_quat(ortho6)
            quats = quats.reshape(T, J, 4)

            eulers = quat.to_euler(
                quats,
                np.tile(dataset.rot_order, (T, 1, 1))
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

            out_gen = os.path.join(out_dir, f"gen_{os.path.basename(filepath).replace(".bvh","")}_{start}.bvh")
            bvh_gen.save(out_gen)
            # write_bvh(template, bvh_frames, out_gen)

            # out_gt = os.path.join(out_dir, f"gt_{os.path.basename(filepath).replace(".bvh","")}_{start}.bvh")
            # write_bvh(template, gt_frames, out_gt)

            print(f"Exported sample {idx} →")
            print(f"  Generated:   {out_gen}")
            # print(f"  GroundTruth: {out_gt}")
            print(f"  Average rollout loss = {avg_loss:.6f}")
            print("")

    print("-------------------------------------------------")
    print(f"Overall Mean loss over {num_samples} samples: {sum(all_losses) / len(all_losses):.6f}")
    print("Done!")