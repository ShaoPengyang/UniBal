import torch
from torch.utils.data import Dataset


class ForgetTopKRetainDataset(Dataset):
    """Wraps forget and retain datasets with dynamic top-k retain selection.

    Instead of randomly sampling from the entire retain set, this dataset
    maintains a mutable index list (typically the top-k highest-loss samples)
    and samples only from that subset. The index list can be updated in-place
    between epochs without recreating the DataLoader.
    """

    def __init__(self, forget, retain, top_k=None):
        self.forget = forget
        self.retain = retain
        self.top_k = top_k if top_k is not None else len(forget)
        self.retain_indices = list(range(len(retain)))

    def update_retain_indices(self, indices):
        """Replace the active retain subset with *indices* (e.g. top-k by loss)."""
        self.retain_indices = indices

    def __len__(self):
        return len(self.forget)

    def __getitem__(self, idx):
        item = {}
        item["forget"] = self.forget[idx]
        pos = torch.randint(0, len(self.retain_indices), (1,)).item()
        item["retain"] = self.retain[self.retain_indices[pos]]
        return item


class ForgetRetainDataset(Dataset):
    # https://github.com/OPTML-Group/SOUL/blob/main/src/dataset/Base.py
    def __init__(self, forget, retain, anchor="forget"):
        """Wraps the forget retain dataset into unlearning dataset.

        Args:
            forget (Dataset): Forget Dataset
            retain (Dataset): Retain Dataset
            anchor (str, optional): Specifies which dataset to anchor while randomly sampling from the other dataset. Defaults to 'forget'.
        """
        self.forget = forget
        self.retain = retain
        self.anchor = anchor

    def __len__(self):
        """Ensures the sampled dataset matches the anchor dataset's length."""
        if self.anchor == "forget":
            assert self.forget is not None, ValueError(
                "forget dataset can't be None when anchor=forget"
            )
            return len(self.forget)
        elif self.anchor == "retain":
            assert self.retain is not None, ValueError(
                "retain dataset can't be None when anchor=retain"
            )
            return len(self.retain)
        else:
            raise NotImplementedError(f"{self.anchor} can be only forget or retain")

    def __getitem__(self, idx):
        item = {}
        if self.anchor == "forget":
            item["forget"] = self.forget[idx]
            if self.retain:
                retain_idx = torch.randint(0, len(self.retain), (1,)).item()
                item["retain"] = self.retain[retain_idx]
        elif self.anchor == "retain":
            item["retain"] = self.retain[idx]
            if self.forget:
                forget_idx = torch.randint(0, len(self.forget), (1,)).item()
                item["forget"] = self.forget[forget_idx]
        return item
