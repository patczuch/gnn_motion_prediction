import os
import torch
from torch_geometric.data import Data, Dataset
from bvh import Bvh
from scipy.spatial.transform import Rotation as R


class BVHMotionDataset(Dataset):
    def __init__(self, directory, context=10, step=1):
        super().__init__()
        self.directory = directory
        self.context = context
        self.total_frames = context + 1
        self.step = step

        self.samples = []
        self.joint_names = None
        self.edges = None

        for fname in os.listdir(directory):
            if not fname.endswith(".bvh"):
                continue
            filepath = os.path.join(directory, fname)

            with open(filepath) as f:
                mocap = Bvh(f.read())

            nframes = mocap.nframes

            for start in range(0, nframes - self.total_frames, self.step):
                self.samples.append((filepath, start))

            if self.joint_names is None:
                self.joint_names = mocap.get_joints_names()
                self.edges = self._build_edges(mocap)

    def _build_edges(self, mocap):
        edges = []
        for j, name in enumerate(mocap.get_joints_names()):
            p = mocap.joint_parent_index(name)
            if p != -1:
                edges.append((p, j))
                edges.append((j, p))  # bidirectional
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def _rotmat(self, mocap, joint, frame):
        ch = mocap.joint_channels(joint)
        if not all(c in ch for c in ["Xrotation","Yrotation","Zrotation"]):
            return torch.eye(3, dtype=torch.float32)

        euler = (
            mocap.frame_joint_channel(frame, joint, "Xrotation"),
            mocap.frame_joint_channel(frame, joint, "Yrotation"),
            mocap.frame_joint_channel(frame, joint, "Zrotation"),
        )
        return torch.tensor(R.from_euler("xyz", euler, degrees=True).as_matrix(),
                            dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, start = self.samples[idx]

        with open(filepath) as f:
            mocap = Bvh(f.read())

        J = len(self.joint_names)
        H = self.context

        context = torch.zeros(J, H, 9)
        for t in range(H):
            frame = start + t
            for j, name in enumerate(self.joint_names):
                context[j, t] = self._rotmat(mocap, name, frame).reshape(9)

        target_frame = start + H
        target = torch.zeros(J, 9)
        for j, name in enumerate(self.joint_names):
            target[j] = self._rotmat(mocap, name, target_frame).reshape(9)

        src_x = context.reshape(J, H * 9)

        batch = torch.zeros(J, dtype=torch.long)
        mask = torch.zeros(J, dtype=torch.bool)

        src_graph = Data()
        src_graph.src_x = src_x
        src_graph.edge_index_bidirection = self.edges
        src_graph.batch = batch
        src_graph.mask = mask

        tgt_graph = Data()
        tgt_graph.tgt_x = target
        tgt_graph.edge_index_bidirection = self.edges
        tgt_graph.batch = batch
        tgt_graph.mask = mask

        return src_graph, tgt_graph
