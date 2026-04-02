from torch.utils.data import Dataset
import torch
import numpy as np


class EEGDataset(Dataset):
    def __init__(self, filename):
        data = np.load(filename)
        X = data['X']  # (n_trials, n_channels, n_timepoints)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (n_trials, 1, n_channels, n_timepoints)
        self.y = torch.tensor(data['y'], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def main():
    dataset = EEGDataset('./data_collection/preprocessed_data/chengyi.npz')
    print(f"Dataset size: {len(dataset)}")
    X, y = dataset[0]
    print(f"X shape: {X.shape}")  # (1, n_channels, n_timepoints)
    print(f"y: {y}")


if __name__ == "__main__":
    main()