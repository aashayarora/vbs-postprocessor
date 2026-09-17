import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class ABCDDataSet(Dataset):
    def __init__(self, data, constraint_data, labels, weights):
        self.data = data
        self.constraint_data = constraint_data
        self.labels = labels
        self.weights = weights

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            self.data[index],
            self.constraint_data[index],
            self.labels[index],
            self.weights[index],
        )

def get_dataloader(dnn_input_data, constraint_data, labels, weights, batch_size, use_sampler=True, is_validation=False):
    dataset = ABCDDataSet(dnn_input_data, constraint_data, labels, weights)

    if use_sampler:
        safe_weights = weights.clone().detach().float()
        safe_weights = torch.clamp(safe_weights, min=0)
        if torch.sum(safe_weights) <= 0:
            safe_weights = torch.ones_like(safe_weights)

        sampler = WeightedRandomSampler(
            weights=safe_weights,
            num_samples=len(safe_weights),
            replacement=True,
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    return DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=not is_validation)