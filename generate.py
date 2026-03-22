import math
import os
import torch
import numpy as np
from torch_geometric.data import Data
from motionpredictor import Model
from pymotion.io.bvh import BVH
import pymotion.rotations.ortho6d as sixd
import pymotion.rotations.ortho6d_torch as sixd_torch
import pymotion.rotations.quat as quat
import config

CHECKPOINT_PATH = "./checkpoints/model_20251228-173213-best.pth"

INPUT_BVH_PATH = "./datasets/lafan1eval/jumps1_subject5.bvh"
INPUT_BONE_LENGTHS_BVH_PATH = "./datasets/lafan1eval/aiming2_subject5.bvh"

START_FRAME = 5900

NUM_FRAMES_TO_GENERATE = 100

OUTPUT_DIR = "./gen_results"

OUTPUT_NAME = "gen_"+INPUT_BVH_PATH[INPUT_BVH_PATH.rfind("/")+1: INPUT_BVH_PATH.rfind(".")] + "_" + str(START_FRAME) + "_" + str(NUM_FRAMES_TO_GENERATE)

def load_bvh_features(bvh_path):
    bvh = BVH()
    bvh.load(bvh_path)

    rotations_quat, local_positions, parents, offsets, end_sites, end_sites_parents = bvh.get_data()

    rot6_np = sixd.from_quat(rotations_quat)  # (F, J, 3, 2)
    rot6 = torch.from_numpy(rot6_np).float()
    rot6 = rot6.reshape(rot6.shape[0], rot6.shape[1], -1)  # (F, J, 6)

    feats = rot6  # (F, J, 6)

    bone_lengths = torch.from_numpy(
        np.linalg.norm(offsets, axis=1, keepdims=True)
    ).float()  # (J, 1)

    edges = torch.tensor(
        [(parents[j], j) for j in range(1, len(parents))]
    ).t().long()

    bvh_data = {
        "names": bvh.data["names"],
        "offsets": offsets,
        "parents": parents,
        "end_sites": end_sites,
        "end_sites_parents": end_sites_parents,
        "rot_order": bvh.data["rot_order"],
        "frame_time": bvh.data["frame_time"],
    }

    return feats, bone_lengths, edges, bvh_data


def create_graph(context_rotations, bone_lengths, edges, device):
    context_length, J, rotation_dim = context_rotations.shape

    context_flat = context_rotations.permute(1, 0, 2).reshape(J, context_length * rotation_dim)

    x = torch.cat([bone_lengths, context_flat], dim=1)  # (J, 1 + context_length * rotation_dim)

    batch = torch.zeros(J, dtype=torch.long, device=device)

    return Data(x=x.to(device), edge_index=edges.to(device), batch=batch)


def create_lastframe_graph(last_frame_rotation, bone_lengths, edges, device):
    J = last_frame_rotation.shape[0]

    x = torch.cat([bone_lengths, last_frame_rotation], dim=1)  # (J, 1 + rotation_dim)

    batch = torch.zeros(J, dtype=torch.long, device=device)

    return Data(x=x.to(device), edge_index=edges.to(device), batch=batch)


def normalize_rotation(rot6):
    rot_mat = sixd_torch.to_matrix(rot6.view(-1, 3, 2))  # (J, 3, 3)
    rot6_norm = rot_mat[..., :3, :2].reshape(rot6.shape[0], -1)
    return rot6_norm


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    context_length = config.context_length
    gen_frames = config.gen_frames
    rotation_dim = config.rotation_dim
    bone_length_dim = config.bone_length_dim

    num_iterations = math.ceil(NUM_FRAMES_TO_GENERATE / gen_frames)
    actual_generated_frames = num_iterations * gen_frames

    print(f"Context length: {context_length}")
    print(f"Model generates {gen_frames} frames per iteration")
    print(f"Requested frames: {NUM_FRAMES_TO_GENERATE}")
    print(f"Will generate {actual_generated_frames} frames in {num_iterations} iterations")

    print(f"\nLoading BVH file: {INPUT_BVH_PATH}")
    feats, _, edges, bvh_data = load_bvh_features(INPUT_BVH_PATH)
    _, bone_lengths, _, _ = load_bvh_features(INPUT_BONE_LENGTHS_BVH_PATH)

    F, J, _ = feats.shape
    print(f"BVH has {F} frames, {J} joints")

    if START_FRAME + context_length > F:
        raise ValueError(
            f"Start frame {START_FRAME} + context {context_length} exceeds "
            f"total frames {F} in BVH file"
        )

    print(f"\nLoading model: {CHECKPOINT_PATH}")
    model = Model().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    context_rotations = feats[START_FRAME:START_FRAME + context_length].clone()  # (context_length, J, rotation_dim)

    generated_frames = []

    context_frames_for_output = [
        feats[START_FRAME + f, :, :rotation_dim].clone()
        for f in range(context_length)
    ]

    print(f"\nGenerating frames starting from frame {START_FRAME}...")

    with torch.no_grad():
        for iteration in range(num_iterations):
            print(f"  Iteration {iteration + 1}/{num_iterations}")

            src_graph = create_graph(context_rotations, bone_lengths, edges, device)

            last_frame = context_rotations[-1]  # (J, rotation_dim)
            lastframe_graph = create_lastframe_graph(last_frame, bone_lengths, edges, device)

            pred = model(src_graph, lastframe_graph)  # (J, gen_frames * rotation_dim)
            pred_seq = pred.view(J, gen_frames, rotation_dim)  # (J, gen_frames, rotation_dim)

            for step in range(gen_frames):
                pred_step = pred_seq[:, step, :]  # (J, rotation_dim)

                pred_norm = normalize_rotation(pred_step)

                generated_frames.append(pred_norm.cpu())

            new_context = torch.zeros_like(context_rotations)
            new_context[:-gen_frames] = context_rotations[gen_frames:]

            for step in range(gen_frames):
                pred_step = pred_seq[:, step, :]
                pred_norm = normalize_rotation(pred_step)
                new_context[-(gen_frames - step)] = pred_norm.cpu()

            context_rotations = new_context

    print(f"\nGenerated {len(generated_frames)} frames")

    all_frames = context_frames_for_output + generated_frames
    total_frames = len(all_frames)

    print(f"Total frames (context + generated): {total_frames}")

    bvh_rots = torch.stack(all_frames)  # (total_frames, J, rotation_dim)

    ortho6 = bvh_rots.view(-1, 3, 2).cpu().numpy()
    quats = sixd.to_quat(ortho6)  # (total_frames * J, 4)
    quats = quats.reshape(total_frames, J, 4)  # (total_frames, J, 4)

    eulers = quat.to_euler(
        quats,
        np.tile(bvh_data["rot_order"], (total_frames, 1, 1)),
    ) * 180 / math.pi

    bvh_out = BVH()
    bvh_out.data = {
        "names": bvh_data["names"],
        "offsets": bvh_data["offsets"],
        "end_sites": bvh_data["end_sites"],
        "end_sites_parents": bvh_data["end_sites_parents"],
        "parents": bvh_data["parents"],
        "rot_order": bvh_data["rot_order"],
        "positions": np.zeros((total_frames, len(bvh_data["names"]), 3)),
        "rotations": eulers,
        "frame_time": bvh_data["frame_time"],
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}.bvh")
    bvh_out.save(output_path)

    print(f"\nSaved generated motion to: {output_path}")
    print(f"  Context frames: {context_length} (frames {START_FRAME} to {START_FRAME + context_length - 1})")
    print(f"  Generated frames: {actual_generated_frames}")
    print(f"  Total frames in output: {total_frames}")
    print("Done!")


if __name__ == "__main__":
    main()

