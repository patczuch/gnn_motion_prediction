from torch.utils.data import Dataset


class MotionDataset(Dataset):
    def __init__(self, sequences, seq_len=10):
        self.seq_len = seq_len
        self.sequences = sequences  # list of (T, num_joints, 15) — [root(3) + rot6d(6) + maybe others]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_seq = seq[:self.seq_len]
        target = seq[self.seq_len]
        return input_seq, target