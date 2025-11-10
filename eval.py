import os
import random
import torch
from motionpredictor import Model
from dataset import BVHMotionDataset
from rot_utils import sixd_to_matrix, sixd_to_euler, euler_to_sixd
from bvh_utils import write_bvh
from geodesicloss import GeodesicLoss

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    checkpoint_path = "./checkpoints/model_20251110-161657.pth"
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

    loss_fn = GeodesicLoss(reduction='mean')

    print(f"Loaded dataset with {len(dataset)} samples")
    print(f"Testing model: {checkpoint_path}")
    print(f"Rollout steps: {rollout}")

    test_indices = random.sample(range(len(dataset)), num_samples)
    all_losses = []

    with torch.no_grad():
        for idx in test_indices:
            src_graph, lastframe_graph, tgt_graph = dataset[idx]

            src_graph = src_graph.to(device)
            lastframe_graph = lastframe_graph.to(device)

            J = src_graph.x.shape[0]
            context_tensor = src_graph.x.view(J, context, 6)

            filepath, start = dataset.samples[idx]
            rot = dataset.cache[filepath]

            gen_frames = [rot[start + f] for f in range(context)]

            losses = []

            for s in range(rollout):
                src_graph.x = context_tensor.reshape(J, context * 6).to(device)
                pred = model(src_graph, lastframe_graph)
                pred = euler_to_sixd(sixd_to_euler(pred))
                gen_frames.append(pred.cpu())

                gt = rot[start + context + s]

                loss = loss_fn(sixd_to_matrix(pred).cpu(), sixd_to_matrix(gt).cpu()).item()
                losses.append(loss)
                print(f"Sample {idx} frame {s} | Loss = {loss:.6f}")

                context_tensor = torch.cat([context_tensor[:, 1:, :], pred.view(J, 1, 6).detach().to(context_tensor.device)], dim=1)
                lastframe_graph.x = pred.to(device)

            avg_loss = sum(losses) / len(losses)
            all_losses.append(avg_loss)

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
            print(f"  Average rollout loss = {avg_loss:.6f}")
            print("")

    print("-------------------------------------------------")
    print(f"Overall Mean loss over {num_samples} samples: {sum(all_losses) / len(all_losses):.6f}")
    print("Done!")