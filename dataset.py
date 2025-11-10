import os
import torch
from torch_geometric.data import Data, Dataset
from bvh import Bvh
from rot_utils import euler_to_sixd


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

            with open(filepath) as f:
                mocap = Bvh(f.read())

            if not hasattr(self, "joint_names"):
                self.joint_names = mocap.get_joints_names()
                self.edges = self._build_edges(mocap)

            J = len(self.joint_names)
            F = mocap.nframes

            angles_deg = torch.zeros(F, J, 3)

            if os.path.exists(cache_path):
                rotmats = torch.load(cache_path)
                print("  loaded precomputed rotations")
            else:
                for j, name in enumerate(self.joint_names):
                    for ci, ch in enumerate(["Xrotation","Yrotation","Zrotation"]):
                        values = [mocap.frame_joint_channel(f, name, ch)
                                  for f in range(F)]
                        angles_deg[:,j,ci] = torch.tensor(values)

                rotmats = euler_to_sixd(angles_deg)

                torch.save(rotmats, cache_path)
                print("  wrote cached rotations")

            self.cache[filepath] = rotmats
            F = rotmats.shape[0]

            for start in range(0, F - self.total_frames, self.step):
                self.samples.append((filepath, start))

        print(f"Dataset ready: {len(self.samples)} samples")

    def _build_edges(self, mocap):
        edges = []
        for j, name in enumerate(mocap.get_joints_names()):
            p = mocap.joint_parent_index(name)
            if p != -1:
                edges.append((p, j))
                edges.append((j, p))
        return torch.tensor(edges).t().contiguous()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, start = self.samples[idx]
        rot = self.cache[filepath]

        J = rot.shape[1]
        H = self.context

        rotsize = 6
        context = rot[start:start+H].reshape(H, J, rotsize).permute(1,0,2).reshape(J, H*rotsize)
        target  = rot[start+H].reshape(J, rotsize)
        lastframe  = rot[start+H-1].reshape(J, rotsize)
        # lastframe = torch.zeros(target.shape)

        batch = torch.zeros(J, dtype=torch.long)

        src_graph = Data(x=context, edge_index=self.edges, batch=batch)
        lastframe_graph = Data(x=lastframe, edge_index=self.edges, batch=batch)
        tgt_graph = Data(x=target,  edge_index=self.edges, batch=batch)

        return src_graph, lastframe_graph, tgt_graph