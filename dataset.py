import os
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from pymotion.io.bvh import BVH
import pymotion.rotations.ortho6d as sixd

import config


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
        self.skeleton_cache = {}
        self._skel_key_to_id = {}
        self.get_cache = []

        print("Loading dataset...")
        for directory in directories:
            bvh_files = [f for f in os.listdir(directory) if f.endswith(".bvh")]

            for i, fname in enumerate(bvh_files, 1):
                print(f"{fname} ({i}/{len(bvh_files)})")

                filepath = os.path.join(directory, fname)
                cache_path = filepath + ".feat.pt"

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
                        continue
                if flag:
                    continue

                # Create undirected graph (bidirectional edges)
                edges = torch.tensor(
                    [(parents[j], j) for j in range(1, len(parents))] +
                    [(j, parents[j]) for j in range(1, len(parents))]
                ).t().long()
                bone_lengths = torch.from_numpy(
                    np.linalg.norm(offsets, axis=1, keepdims=True)
                ).float()

                J = parents.shape[0]
                ancestor_counts = torch.zeros(J, dtype=torch.float32)

                for j in range(J):
                    p = parents[j].item()
                    while p != 0:
                        ancestor_counts[j] += 1
                        p = parents[p].item()

                joint_weights = 1.0 / (1.0 + ancestor_counts)  # shape [J]
                joint_weights = joint_weights.view(1, 1, J, 1)  # reshape for broadcasting

                self.skeleton_cache[filepath] = {
                    "offsets": offsets,
                    "parents": parents,
                    "end_sites": end_sites,
                    "end_sites_parents": end_sites_parents,
                    "names": bvh.data["names"],
                    "rot_order": bvh.data["rot_order"],
                    "frame_time": bvh.data["frame_time"],
                    "edges": edges,
                    "bone_lengths": bone_lengths,
                    "joint_weights": joint_weights
                }

                skel_key = (len(parents), tuple(parents))
                if skel_key not in self._skel_key_to_id:
                    self._skel_key_to_id[skel_key] = len(self._skel_key_to_id)
                skel_id = self._skel_key_to_id[skel_key]

                if os.path.exists(cache_path):
                    feats = torch.load(cache_path)
                    print("  loaded precomputed features")
                else:
                    rot6_np = sixd.from_quat(rotations_quat)
                    rot6 = torch.from_numpy(rot6_np).float()
                    rot6 = rot6.reshape(rot6.shape[0], rot6.shape[1], -1)

                    feats = torch.cat([rot6], dim=-1)

                self.cache[filepath] = feats

                # Cache root positions: (F, 3)
                root_positions = torch.from_numpy(
                    local_positions[:, 0, :].copy()
                ).float()
                self.root_pos_cache[filepath] = root_positions

                F = feats.shape[0]

                for start in range(0, F - (self.context + config.gen_frames), self.step):
                    self.samples.append((filepath, start))
                    self.sample_skel.append(skel_id)

        self.sample_skel = np.array(self.sample_skel)
        self.get_cache = [None for _ in range(len(self.samples))]
        print(f"Dataset ready: {len(self.samples)} samples, {len(self._skel_key_to_id)} unique skeletons")

    def __len__(self):
        return len(self.samples)

    def get_skeleton(self, filepath):
        return self.skeleton_cache[filepath]

    def get_skeleton_id(self, idx):
        return self.sample_skel[idx]

    def __getitem__(self, idx):
        if self.get_cache[idx] is not None:
            return self.get_cache[idx]

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

        context = full_window[:H].permute(1, 0, 2).reshape(J, H * rotsize)
        target = full_window[H:H + gen_frames].permute(1, 0, 2).reshape(J, gen_frames * rotsize)

        context = torch.cat([context], dim=1)
        target = torch.cat([target], dim=1)

        # Root positions for context and target
        root_pos_window = root_positions[start:start + H + gen_frames]  # (H+gen_frames, 3)
        root_pos_origin = root_pos_window[0:1]  # (1, 3) — first context frame
        root_pos_window = root_pos_window - root_pos_origin  # delta from first frame

        root_pos_context = root_pos_window[:H].reshape(H * 3)           # (H*3,)
        root_pos_target = root_pos_window[H:H + gen_frames].reshape(gen_frames * 3)  # (gen_frames*3,)

        batch = torch.zeros(J, dtype=torch.long)

        parents_t = torch.tensor(skeleton["parents"], dtype=torch.long)
        offsets_t = torch.from_numpy(skeleton["offsets"]).float()

        src_graph = Data(
            x=context, edge_index=edges, batch=batch,
            parents=parents_t, offsets=offsets_t, joint_weights=skeleton["joint_weights"],
            root_pos=root_pos_context,
        )
        tgt_graph = Data(
            x=target, edge_index=edges, batch=batch,
            root_pos=root_pos_target,
        )

        self.get_cache[idx] = (src_graph, tgt_graph)
        return src_graph, tgt_graph
