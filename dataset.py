import os
import torch
from torch_geometric.data import Data, Dataset
from bvh import Bvh


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

                angles = angles_deg * 3.14159265359 / 180.0
                rotmats = euler_to_matrix_xyz(angles)

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

        context = rot[start:start+H].reshape(H, J, 9).permute(1,0,2).reshape(J, H*9)
        target  = rot[start+H].reshape(J, 9)

        batch = torch.zeros(J, dtype=torch.long)

        src_graph = Data(x=context, edge_index=self.edges, batch=batch)
        tgt_graph = Data(x=target,  edge_index=self.edges, batch=batch)

        return src_graph, tgt_graph

def euler_to_matrix_xyz(e):
    x, y, z = e[...,0], e[...,1], e[...,2]

    cx, cy, cz = torch.cos(x), torch.cos(y), torch.cos(z)
    sx, sy, sz = torch.sin(x), torch.sin(y), torch.sin(z)

    rot = torch.zeros(e.shape[:-1] + (3,3), dtype=e.dtype)

    rot[...,0,0] = cy*cz
    rot[...,0,1] = -cy*sz
    rot[...,0,2] = sy

    rot[...,1,0] = sx*sy*cz + cx*sz
    rot[...,1,1] = -sx*sy*sz + cx*cz
    rot[...,1,2] = -sx*cy

    rot[...,2,0] = -cx*sy*cz + sx*sz
    rot[...,2,1] = cx*sy*sz + sx*cz
    rot[...,2,2] = cx*cy

    return rot