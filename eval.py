import os
import random
import torch
from torch import nn
from motionpredictor import Model
from dataset import BVHMotionDataset


def write_bvh(template_lines, rotations, out_path):
    out = []
    i = 0
    while i < len(template_lines):
        line = template_lines[i]
        out.append(line)
        if line.startswith("Frame Time"):
            i += 1
            break
        i += 1

    F, J, _, _ = rotations.shape

    for f in range(F):
        out.append("0.0 0.0 0.0 ")
        e = matrix_to_euler_zyx(rotations[f])
        e = torch.rad2deg(e)

        frame_euler = []

        for j in range(J):
            x, y, z = e[j]
            frame_euler.extend([z.item(), y.item(), x.item()])

        out.append(" ".join(f"{v:.6f}" for v in frame_euler) + "\n")

    with open(out_path, "w") as f:
        f.writelines(out)



def matrix_to_euler_zyx(R):
    r20 = R[..., 2, 0]
    cy = torch.sqrt(R[..., 0, 0]**2 + R[..., 1, 0]**2)

    y = torch.asin(torch.clamp(-r20, -1.0, 1.0))

    eps = 1e-6
    x = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    z = torch.atan2(R[..., 1, 0], R[..., 0, 0])

    mask = cy < eps
    if mask.any():
        x_alt = torch.atan2(-R[..., 0, 1], R[..., 1, 1])
        x = torch.where(mask, x_alt, x)
        z = torch.where(mask, torch.zeros_like(z), z)

    e = torch.stack((x, y, z), dim=-1)
    return e


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    checkpoint_path = "./checkpoints/model_20251102-155319.pth"
    dataset_path = "datasets/lafan1eval"
    out_dir = "./eval_results"
    os.makedirs(out_dir, exist_ok=True)

    context = 10
    rollout = 5
    num_samples = 5
    random.seed(10)

    dataset = BVHMotionDataset(dataset_path, context=context, step=10)

    model = Model().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    loss_fn = nn.MSELoss(reduction='sum')

    print(f"Loaded dataset with {len(dataset)} samples")
    print(f"Testing model: {checkpoint_path}")
    print(f"Rollout steps: {rollout}")

    test_indices = random.sample(range(len(dataset)), num_samples)
    all_mse_scores = []

    with torch.no_grad():
        for idx in test_indices:
            src_graph, tgt_graph = dataset[idx]

            src_graph = src_graph.to(device)
            tgt_graph = tgt_graph.to(device)

            J = src_graph.x.shape[0]
            context_tensor = src_graph.x.view(J, context, 9)

            filepath, start = dataset.samples[idx]
            rot = dataset.cache[filepath]

            gen_frames = [rot[start + f] for f in range(context)]

            mse_scores = []

            for s in range(rollout):
                src_graph.x = context_tensor.reshape(J, context * 9)
                _, pred = model(src_graph, tgt_graph)

                pred_rot = pred.view(J, 3, 3)

                gen_frames.append(pred_rot.cpu())

                gt_rot = rot[start + context + s]
                tgt_graph.x = gt_rot.reshape(J, 9).to(device)

                mse = loss_fn(pred_rot.cpu(), gt_rot.cpu()).item()
                mse_scores.append(mse)
                print(f"Sample {idx} frame {s} | MSE = {mse:.6f}")

                pred_frame_flat = pred.view(J, 1, 9)
                context_tensor = torch.cat([context_tensor[:,1:,:],
                                            pred_frame_flat], dim=1)

            avg_mse = sum(mse_scores) / len(mse_scores)
            all_mse_scores.append(avg_mse)

            gen_frames = torch.stack(gen_frames)

            gt_frames = rot[start:start+context+rollout]

            with open(filepath, "r") as f:
                template = f.readlines()

            out_gen = os.path.join(out_dir, f"gen_{os.path.basename(filepath).replace(".bvh","")}_{start}.bvh")
            write_bvh(template, gen_frames, out_gen)

            out_gt = os.path.join(out_dir, f"gt_{os.path.basename(filepath).replace(".bvh","")}_{start}.bvh")
            write_bvh(template, gt_frames, out_gt)

            print(f"Exported sample {idx} →")
            print(f"  Generated:   {out_gen}")
            print(f"  GroundTruth: {out_gt}")
            print(f"  Average rollout MSE = {avg_mse:.6f}")
            print("")

    print("-------------------------------------------------")
    print(f"Overall Mean MSE over {num_samples} samples: {sum(all_mse_scores) / len(all_mse_scores):.6f}")
    print("Done!")