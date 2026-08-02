import os
import random
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from pymotion.io.bvh import BVH
import pymotion.ops.skeleton as skeleton_ops
import pymotion.rotations.ortho6d_torch as sixd_torch
import pymotion.rotations.quat_torch as quat_torch

import config


ROT_JUMP_THRESH = 90.0
POS_JUMP_THRESH = 25.0

UP_AXIS = 1


def yaw_rotmat(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([
        torch.stack([c, z, s], dim=-1),
        torch.stack([z, o, z], dim=-1),
        torch.stack([-s, z, c], dim=-1),
    ], dim=-2)


def root_yaw(rot6_root):
    R = sixd_torch.to_matrix(rot6_root.reshape(-1, 3, 2))
    q = quat_torch.from_matrix(R)  # (F, 4) as [w, x, y, z]
    w, y = q[..., 0], q[..., 2]
    yaw = 2.0 * torch.atan2(y, w)

    ok = (w * w + y * y) > 1e-6
    if not bool(ok.all()):
        order = torch.arange(yaw.shape[0], device=yaw.device)
        src = torch.cummax(torch.where(ok, order, torch.full_like(order, -1)), dim=0).values
        yaw = torch.where(src >= 0, yaw[src.clamp(min=0)], torch.zeros_like(yaw))
    return yaw


def rotate_rot6(rot6, R):
    M = rot6.reshape(rot6.shape[:-1] + (3, 2))
    return torch.einsum('ij,...jk->...ik', R, M).reshape(rot6.shape)


def rotate_vec(v, R):
    return torch.einsum('ij,...j->...i', R, v)


# unit rest-offset direction (3) + bone length relative to the skeleton (1)
GEOM_DIM = 4


def skeleton_geometry_features(offsets):
    off = torch.as_tensor(np.asarray(offsets), dtype=torch.float32)
    lengths = off.norm(dim=-1)

    dirs = off / lengths.clamp(min=1e-8).unsqueeze(-1)
    dirs[lengths < 1e-8] = 0.0
    dirs[0] = 0.0

    lengths = lengths.clone()
    lengths[0] = 0.0
    rel = lengths / lengths[1:].mean().clamp(min=1e-8)

    return torch.cat([dirs, rel.unsqueeze(-1)], dim=-1)


def _quat_to_rotmat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.stack([
        1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w),
        2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w),
        2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y),
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))
    return R


def _geodesic_deg_batch(Ra, Rb):
    R_diff = np.einsum('...ji,...jk->...ik', Ra, Rb)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def compute_bad_transitions(rotations_quat, root_positions,
                             rot_thresh=ROT_JUMP_THRESH,
                             pos_thresh=POS_JUMP_THRESH):
    bad = set()
    F = rotations_quat.shape[0]
    if F < 2:
        return bad

    pos_diff = np.linalg.norm(
        root_positions[1:] - root_positions[:-1], axis=-1
    )  # (F-1,)
    bad_pos = np.where(pos_diff > pos_thresh)[0] + 1
    bad.update(bad_pos.tolist())

    Ra = _quat_to_rotmat(rotations_quat[:-1])   # (F-1, J, 3, 3)
    Rb = _quat_to_rotmat(rotations_quat[1:])    # (F-1, J, 3, 3)
    geo = _geodesic_deg_batch(Ra, Rb)           # (F-1, J)
    max_geo = geo.max(axis=-1)                  # (F-1,)
    bad_rot = np.where(max_geo > rot_thresh)[0] + 1
    bad.update(bad_rot.tolist())

    return bad


def window_has_jump(bad_transitions, start, length):
    for f in bad_transitions:
        if start < f <= start + length - 1:
            return True
    return False


def split_train_val(dataset, val_ratio, seed, block_frames=None):
    window_len = dataset.context + config.gen_frames
    if block_frames is None:
        block_frames = window_len * 8

    blocks = {}
    dropped = 0
    for idx, (filepath, start) in enumerate(dataset.samples):
        block = start // block_frames
        if (start + window_len - 1) // block_frames != block:
            dropped += 1  # straddles a boundary; dropping it creates the gap
            continue
        blocks.setdefault((filepath, block), []).append(idx)

    keys = sorted(blocks.keys())
    random.Random(seed).shuffle(keys)

    target = sum(len(blocks[k]) for k in keys) * val_ratio
    train_indices, val_indices = [], []
    n_val = 0
    for k in keys:
        size = len(blocks[k])
        if abs(n_val + size - target) < abs(n_val - target):
            val_indices.extend(blocks[k])
            n_val += size
        else:
            train_indices.extend(blocks[k])

    train_indices.sort()
    val_indices.sort()
    return train_indices, val_indices, dropped


class BVHMotionDataset(Dataset):
    def __init__(self, directories, context, step):
        super().__init__()
        self.directories = directories
        self.context = context
        self.step = step

        self.samples = []       # (filepath, start)
        self.sample_skel = []   # skeleton_id per sample
        self.cache = {}
        self.root_pos_cache = {}
        self.yaw_cache = {}
        self.skeleton_cache = {}
        self._skel_key_to_id = {}
        self.facing_normalization = config.facing_normalization
        self.skeleton_geometry = config.skeleton_geometry

        print("Loading dataset...")
        for directory in directories:
            bvh_files = [f for f in os.listdir(directory) if f.endswith(".bvh")]

            for i, fname in enumerate(bvh_files, 1):
                print(f"{fname} ({i}/{len(bvh_files)})")

                filepath = os.path.join(directory, fname)

                bvh = BVH()
                bvh.load(filepath)

                rotations_quat, local_positions, parents, offsets, end_sites, end_sites_parents = bvh.get_data()

                if len(rotations_quat) < context + config.gen_frames:
                    print(f"  Skipping {fname} (too short, {len(rotations_quat)} < {context + config.gen_frames})")
                    continue

                flag = False
                for bone_rot_order in bvh.data["rot_order"]:
                    if bone_rot_order[0] != 'x' or bone_rot_order[1] != 'y' or bone_rot_order[2] != 'z':
                        print(f"  Skipping {fname} (unsupported rot_order {bvh.data['rot_order']})")
                        flag = True
                        break
                if flag:
                    continue

                # Create undirected graph (bidirectional edges)
                edges = torch.tensor(
                    [(parents[j], j) for j in range(1, len(parents))] +
                    [(j, parents[j]) for j in range(1, len(parents))]
                ).t().long()
                self.skeleton_cache[filepath] = {
                    "offsets": offsets,
                    "parents": parents,
                    "end_sites": end_sites,
                    "end_sites_parents": end_sites_parents,
                    "names": bvh.data["names"],
                    "rot_order": bvh.data["rot_order"],
                    "frame_time": bvh.data["frame_time"],
                    "edges": edges,
                    "geom": skeleton_geometry_features(offsets),
                }

                skel_key = (len(parents), tuple(parents))
                if skel_key not in self._skel_key_to_id:
                    self._skel_key_to_id[skel_key] = len(self._skel_key_to_id)
                skel_id = self._skel_key_to_id[skel_key]

                cache_path = filepath + ".globalfeat.pt"

                if os.path.exists(cache_path):
                    feats = torch.load(cache_path)
                    print("  loaded precomputed global rotation features")
                else:
                    # Compute global rotations via FK
                    F_total = rotations_quat.shape[0]
                    global_pos_np = np.zeros((F_total, 3), dtype=np.float32)
                    offsets_expanded = np.tile(offsets, (F_total, 1, 1))
                    _, global_rotmats = skeleton_ops.fk(
                        rotations_quat, global_pos_np, offsets_expanded, parents
                    )
                    # global_rotmats: (F, J, 3, 3) -> convert to 6D
                    global_6d = global_rotmats[..., :2]  # (F, J, 3, 2)
                    rot6 = torch.from_numpy(global_6d.copy()).float()
                    rot6 = rot6.reshape(rot6.shape[0], rot6.shape[1], -1)  # (F, J, 6)

                    feats = rot6
                    torch.save(feats, cache_path)
                    print("  computed and cached global rotation features")

                self.cache[filepath] = feats

                self.yaw_cache[filepath] = root_yaw(feats[:, 0, :])

                # Cache root positions: (F, 3)
                root_pos_np = local_positions[:, 0, :].copy()
                root_positions = torch.from_numpy(root_pos_np).float()
                self.root_pos_cache[filepath] = root_positions

                # Compute bad frame transitions (position or rotation jumps)
                bad_transitions = compute_bad_transitions(rotations_quat, root_pos_np)
                if bad_transitions:
                    print(f"  {len(bad_transitions)} bad transition(s) detected, affected windows will be skipped")

                F = feats.shape[0]
                window_len = self.context + config.gen_frames
                skipped = 0

                for start in range(0, F - window_len + 1, self.step):
                    if window_has_jump(bad_transitions, start, window_len):
                        skipped += 1
                        continue
                    self.samples.append((filepath, start))
                    self.sample_skel.append(skel_id)

                if skipped:
                    print(f"  Skipped {skipped} window(s) containing jumps")

        self.sample_skel = np.array(self.sample_skel)
        print(f"Dataset ready: {len(self.samples)} samples, {len(self._skel_key_to_id)} unique skeletons")

    def __len__(self):
        return len(self.samples)

    def get_skeleton(self, filepath):
        return self.skeleton_cache[filepath]

    def get_skeleton_id(self, idx):
        return self.sample_skel[idx]

    def __getitem__(self, idx):
        filepath, start = self.samples[idx]
        feats = self.cache[filepath]
        root_positions = self.root_pos_cache[filepath]
        skeleton = self.skeleton_cache[filepath]
        edges = skeleton["edges"]

        J = feats.shape[1]
        H = self.context
        rotsize = config.rotation_dim

        gen_frames = config.gen_frames

        full_window = feats[start:start + H + gen_frames].reshape(-1, J, rotsize)

        # Root positions for context and target
        root_pos_window = root_positions[start:start + H + gen_frames]  # (H+gen_frames, 3)
        root_pos_origin = root_pos_window[0:1]  # (1, 3) — first context frame
        root_pos_window = root_pos_window - root_pos_origin  # delta from first frame

        # Facing normalization
        if self.facing_normalization:
            R_inv = yaw_rotmat(-self.yaw_cache[filepath][start + H - 1])
            full_window = rotate_rot6(full_window, R_inv)
            root_pos_window = rotate_vec(root_pos_window, R_inv)

        context = full_window[:H].permute(1, 0, 2).reshape(J, H * rotsize)
        target = full_window[H:H + gen_frames].permute(1, 0, 2).reshape(J, gen_frames * rotsize)

        if self.skeleton_geometry:
            context = torch.cat([context, skeleton["geom"]], dim=1)

        root_pos_context = root_pos_window[:H].reshape(H * 3)           # (H*3,)
        root_pos_target = root_pos_window[H:H + gen_frames].reshape(gen_frames * 3)  # (gen_frames*3,)

        batch = torch.zeros(J, dtype=torch.long)

        parents_t = torch.tensor(skeleton["parents"], dtype=torch.long)
        offsets_t = torch.from_numpy(skeleton["offsets"]).float()

        src_graph = Data(
            x=context, edge_index=edges, batch=batch,
            parents=parents_t, offsets=offsets_t,
            root_pos=root_pos_context,
        )
        tgt_graph = Data(
            x=target, edge_index=edges, batch=batch,
            root_pos=root_pos_target,
        )

        return src_graph, tgt_graph
