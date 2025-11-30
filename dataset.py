import os
import torch
from torch_geometric.data import Data, Dataset
from pymotion.io.bvh import BVH
from pymotion.ops.skeleton import fk
import pymotion.rotations.ortho6d as sixd


class BVHMotionDataset(Dataset):
    def __init__(self, directory, context, step):
        super().__init__()
        self.directory = directory
        self.context = context
        self.total_frames = context + 1
        self.step = step

        self.samples = []
        self.cache = {}

        print("Loading dataset...")
        bvh_files = [f for f in os.listdir(directory) if f.endswith(".bvh")]

        for i, fname in enumerate(bvh_files, 1):
            print(f"{fname} ({i}/{len(bvh_files)})")

            filepath = os.path.join(directory, fname)
            cache_path = filepath + ".rot.pt"

            bvh = BVH()
            bvh.load(filepath)

            rotations_quat, local_positions, parents, offsets, end_sites, end_sites_parents = bvh.get_data()

            if not hasattr(self, "names"):
                self.offsets = offsets
                self.parents = parents
                self.end_sites = end_sites
                self.end_sites_parents = end_sites_parents
                self.names = bvh.data["names"]
                self.rot_order = bvh.data["rot_order"]
                self.frame_time = bvh.data["frame_time"]
                self.edges = torch.tensor([(parents[i], i) for i in range(1, len(parents))]).t().long()

            if os.path.exists(cache_path):
                rotmats = torch.load(cache_path)
                print("  loaded precomputed rotations")
            else:
                rotmats = torch.from_numpy(sixd.from_quat(rotations_quat)).float()
                torch.save(rotmats, cache_path)
                print("  wrote cached rotations")

            self.cache[filepath] = rotmats
            F = rotmats.shape[0]

            for start in range(0, F - (self.context + self.context), self.step):
                self.samples.append((filepath, start))

        print(f"Dataset ready: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, start = self.samples[idx]
        rot = self.cache[filepath]

        J = rot.shape[1]
        H = self.context
        rotsize = 6

        # deltas = context_frames.clone()
        # deltas[1:] = context_frames[1:] - context_frames[:-1]
        # deltas[0] = 0
        #
        # context = deltas.reshape(H, J, rotsize).permute(1, 0, 2).reshape(J, H * rotsize)

        context = rot[start:start + H].reshape(H, J, rotsize).permute(1, 0, 2).reshape(J, H * rotsize)

        # target = rot[start + H].reshape(J, rotsize)
        target = rot[start + H:start + H + H].reshape(H, J, rotsize).permute(1, 0, 2) .reshape(J, H * rotsize)

        lastframe = rot[start + H - 1].reshape(J, rotsize)

        batch = torch.zeros(J, dtype=torch.long)

        src_graph = Data(x=context, edge_index=self.edges, batch=batch)
        lastframe_graph = Data(x=lastframe, edge_index=self.edges, batch=batch)
        tgt_graph = Data(x=target, edge_index=self.edges, batch=batch)

        return src_graph, lastframe_graph, tgt_graph
