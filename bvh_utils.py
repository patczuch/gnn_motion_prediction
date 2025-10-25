import os

import torch
from bvh import Bvh
from scipy.spatial.transform import Rotation as R


def load_motion_sequences():
    def get_rot_euler(mocap, joint, frame):
        channels = mocap.joint_channels(joint)
        if not all(ch in channels for ch in ["Xrotation", "Yrotation", "Zrotation"]):
            raise Exception(f"No rotation data for bone {joint} on frame {joint}")
        return (
            mocap.frame_joint_channel(frame, joint, "Xrotation"),
            mocap.frame_joint_channel(frame, joint, "Yrotation"),
            mocap.frame_joint_channel(frame, joint, "Zrotation"),
        )

    def euler_rot_to_matrix(rot):
        return R.from_euler("xyz", rot, degrees=True).as_matrix()

    motion_sequences = []
    directory = "./datasets/lafan1"

    for filename in ["aiming1_subject1.bvh"]:
    # for filename in ["aiming1_subject1.bvh", "dance1_subject1.bvh", "fight1_subject2.bvh", "ground1_subject1.bvh"]:
    # for filename in os.listdir(directory):
        if filename.endswith(".bvh"):
            filepath = os.path.join(directory, filename)
            with open(filepath) as f:
                mocap = Bvh(f.read())

            motion_sequence = []

            #for frame in range(mocap.nframes):
            for frame in range(100):
                frame_data = []
                for joint in mocap.get_joints_names():
                    euler = get_rot_euler(mocap, joint, frame)
                    mat = euler_rot_to_matrix(euler)
                    frame_data.append(mat)

                motion_sequence.append(frame_data)

                if frame % 11 == 0 and frame_data:
                    motion_sequences.append(torch.tensor(frame_data, dtype=torch.float32))

    return motion_sequences


def get_bone_hierarchy():
    hierarchy = []
    with open("./datasets/lafan1/aiming1_subject1.bvh") as f:
        mocap = Bvh(f.read())
        i = 0
        for joint in mocap.get_joints_names():
            if mocap.joint_parent_index(joint) != -1:
                hierarchy.append((mocap.joint_parent_index(joint), i))
                hierarchy.append((i, mocap.joint_parent_index(joint)))
            i += 1
    return hierarchy