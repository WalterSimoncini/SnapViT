from torch.utils.data import Dataset


class FastDataset(Dataset):
    """
        Dataset wrapper that pre-applies a transform to the
        input dataset and keeps samples in memory.
    """
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def precompute(self):
        self.samples = [x for x in self.dataset]

    def __len__(self):
        if not hasattr(self, "samples"):
            raise ValueError("The dataset has not been precomputed yet")

        return len(self.samples)

    def __getitem__(self, index):
        if not hasattr(self, "samples"):
            raise ValueError("The dataset has not been precomputed yet")

        return self.samples[index]
