import math
import os
import torch
import numpy as np
from torch_geometric.data import Data
from motionpredictor import Model
from dataset import (root_yaw, yaw_rotmat, rotate_rot6, rotate_vec,
                     skeleton_geometry_features)
from pymotion.io.bvh import BVH
import pymotion.rotations.ortho6d as sixd
import pymotion.rotations.ortho6d_torch as sixd_torch
import pymotion.rotations.quat as quat
import pymotion.ops.skeleton as skeleton_ops
import config

CHECKPOINT_PATH = config.eval_checkpoint_path

INPUT_BVH_PATH = "./data/eval/lafan1_reduced/aiming2_subject5.bvh"

START_FRAME = 50

NUM_FRAMES_TO_GENERATE = 100

OUTPUT_DIR = "./gen_results"

OUTPUT_NAME = "gen_"+INPUT_BVH_PATH[INPUT_BVH_PATH.rfind("/")+1: INPUT_BVH_PATH.rfind(".")] + "_" + str(START_FRAME) + "_" + str(NUM_FRAMES_TO_GENERATE)

def load_bvh_features(bvh_path):
    bvh = BVH()
    bvh.load(bvh_path)

    rotations_quat, local_positions, parents, offsets, end_sites, end_sites_parents = bvh.get_data()

    # Compute global rotations via FK
    F_total = rotations_quat.shape[0]
    global_pos_np = np.zeros((F_total, 3), dtype=np.float32)
    offsets_expanded = np.tile(offsets, (F_total, 1, 1))
    _, global_rotmats = skeleton_ops.fk(
        rotations_quat, global_pos_np, offsets_expanded, parents
    )
    # global_rotmats: (F, J, 3, 3) -> 6D
    global_6d = global_rotmats[..., :2]  # (F, J, 3, 2)
    rot6 = torch.from_numpy(global_6d.copy()).float()
    rot6 = rot6.reshape(rot6.shape[0], rot6.shape[1], -1)  # (F, J, 6)

    feats = rot6  # (F, J, 6) - global rotations

    edges = torch.tensor(
        [(parents[j], j) for j in range(1, len(parents))] +
        [(j, parents[j]) for j in range(1, len(parents))]
    ).t().long()

    # Root positions: (F, 3)
    root_positions = torch.from_numpy(local_positions[:, 0, :].copy()).float()

    bvh_data = {
        "names": bvh.data["names"],
        "offsets": offsets,
        "parents": parents,
        "end_sites": end_sites,
        "end_sites_parents": end_sites_parents,
        "rot_order": bvh.data["rot_order"],
        "frame_time": bvh.data["frame_time"],
    }

    return feats, edges, bvh_data, root_positions


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

    num_iterations = math.ceil(NUM_FRAMES_TO_GENERATE / gen_frames)
    actual_generated_frames = num_iterations * gen_frames

    print(f"Context length: {context_length}")
    print(f"Model generates {gen_frames} frames per iteration")
    print(f"Requested frames: {NUM_FRAMES_TO_GENERATE}")
    print(f"Will generate {actual_generated_frames} frames in {num_iterations} iterations")

    print(f"\nLoading BVH file: {INPUT_BVH_PATH}")
    feats, edges, bvh_data, root_positions = load_bvh_features(INPUT_BVH_PATH)

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

    # Root position handling
    root_pos_origin = root_positions[START_FRAME:START_FRAME + 1]  # (1, 3)
    root_pos_context_delta = root_positions[START_FRAME:START_FRAME + context_length] - root_pos_origin

    generated_frames = []
    generated_root_pos = []  # list of (3,) tensors, delta from origin

    context_frames_for_output = [
        feats[START_FRAME + f, :, :rotation_dim].clone()
        for f in range(context_length)
    ]

    print(f"\nGenerating frames starting from frame {START_FRAME}...")

    with torch.no_grad():
        for iteration in range(num_iterations):
            print(f"  Iteration {iteration + 1}/{num_iterations}")

            # Build root_pos_context for this iteration.
            if iteration == 0:
                window_root = root_pos_context_delta
            else:
                # Use last (context_length) root positions: mix of gt context + generated
                all_root = torch.cat([root_pos_context_delta] + [g.unsqueeze(0) for g in generated_root_pos], dim=0)
                window_root = all_root[-context_length:]

            window_origin = window_root[0:1].clone()  # (1, 3)
            window_root = window_root - window_origin

            ctx_rot_in = context_rotations
            if config.facing_normalization:
                theta = root_yaw(context_rotations[-1, 0:1, :])[0]
                R_inv, R_fwd = yaw_rotmat(-theta), yaw_rotmat(theta)
                ctx_rot_in = rotate_rot6(context_rotations, R_inv)
                window_root = rotate_vec(window_root, R_inv)

            rpc = window_root.reshape(context_length * 3)

            x_input = ctx_rot_in.permute(1, 0, 2).reshape(J, context_length * rotation_dim)
            if config.skeleton_geometry:
                x_input = torch.cat(
                    [x_input, skeleton_geometry_features(bvh_data["offsets"])], dim=1
                )
            batch = torch.zeros(J, dtype=torch.long, device=device)
            parents_t = torch.tensor(bvh_data["parents"], dtype=torch.long)
            offsets_t = torch.from_numpy(bvh_data["offsets"]).float()

            src_graph = Data(
                x=x_input.to(device), edge_index=edges.to(device), batch=batch,
                parents=parents_t, offsets=offsets_t,
                root_pos=rpc.to(device),
            )

            pred_rot, pred_root_pos = model(src_graph)
            pred_seq = pred_rot.view(J, gen_frames, rotation_dim).cpu()
            pred_root_pos_delta = pred_root_pos.view(gen_frames, 3).cpu()

            if config.facing_normalization:
                pred_seq = rotate_rot6(pred_seq, R_fwd)
                pred_root_pos_delta = rotate_vec(pred_root_pos_delta, R_fwd)

            pred_root_pos_delta = pred_root_pos_delta + window_origin

            for step in range(gen_frames):
                pred_step = pred_seq[:, step, :]
                pred_norm = normalize_rotation(pred_step)
                generated_frames.append(pred_norm.cpu())
                generated_root_pos.append(pred_root_pos_delta[step])

            # Shift context window
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

    # Convert global 6D -> global quats -> local quats for BVH export
    ortho6 = bvh_rots.view(-1, 3, 2).cpu().numpy()
    global_quats = sixd.to_quat(ortho6).reshape(total_frames, J, 4)
    local_quats = skeleton_ops.from_global_rotations(global_quats, bvh_data["parents"])

    eulers = quat.to_euler(
        local_quats,
        np.tile(bvh_data["rot_order"], (total_frames, 1, 1)),
    ) * 180 / math.pi

    # Build root positions for BVH export
    ctx_root_abs = root_positions[START_FRAME:START_FRAME + context_length].numpy()
    gen_root_abs = torch.stack(generated_root_pos).numpy() + root_pos_origin.numpy()  # (gen_frames_total, 3)

    bvh_positions = np.zeros((total_frames, len(bvh_data["names"]), 3))
    bvh_positions[:context_length, 0, :] = ctx_root_abs
    bvh_positions[context_length:, 0, :] = gen_root_abs

    bvh_out = BVH()
    bvh_out.data = {
        "names": bvh_data["names"],
        "offsets": bvh_data["offsets"],
        "end_sites": bvh_data["end_sites"],
        "end_sites_parents": bvh_data["end_sites_parents"],
        "parents": bvh_data["parents"],
        "rot_order": bvh_data["rot_order"],
        "positions": bvh_positions,
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

