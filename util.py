import json
import pickle
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def load_adjacency(path):
    with open(path, "rb") as stream:
        try:
            value = pickle.load(stream)
        except UnicodeDecodeError:
            stream.seek(0)
            value = pickle.load(stream, encoding="latin1")
    if isinstance(value, tuple) and len(value) == 3:
        value = value[2]
    adjacency = np.asarray(value, dtype=np.float32)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    adjacency = adjacency.copy()
    adjacency[adjacency < 0] = 0
    return adjacency


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = float(mean)
        self.std = float(std)

    def transform(self, values):
        return (values - self.mean) / self.std

    def inverse_transform(self, values):
        return values * self.std + self.mean


def metric_mask(labels, null_value=None):
    if null_value is None:
        return torch.ones_like(labels, dtype=torch.bool)
    if np.isnan(null_value):
        return ~torch.isnan(labels)
    return labels != null_value


def metrics(predictions, labels, null_value=None):
    mask = metric_mask(labels, null_value)
    errors = predictions[mask] - labels[mask]
    targets = labels[mask]
    if errors.numel() == 0:
        raise ValueError("no valid targets remain after masking")
    mae = errors.abs().mean()
    rmse = errors.square().mean().sqrt()
    nonzero = targets != 0
    mape = (errors[nonzero].abs() / targets[nonzero].abs()).mean() * 100 if nonzero.any() else torch.full((), float("nan"), device=errors.device)
    return {"mae": mae, "rmse": rmse, "mape": mape}


class ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.as_tensor(inputs, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


def load_datasets(directory, batch_size, seed):
    arrays = {split: np.load(f"{directory}/{split}.npz") for split in ("train", "val", "test")}
    train_x = arrays["train"]["x"].astype(np.float32)
    mean = train_x[..., 0].mean()
    std = train_x[..., 0].std()
    if std == 0:
        raise ValueError("training input standard deviation is zero")
    scaler = StandardScaler(mean, std)
    loaders = {}
    generator = torch.Generator().manual_seed(seed)
    for split, archive in arrays.items():
        inputs = archive["x"].astype(np.float32)
        targets = archive["y"].astype(np.float32)
        inputs[..., 0] = scaler.transform(inputs[..., 0])
        dataset = ArrayDataset(inputs, targets)
        loaders[split] = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=split == "train", generator=generator)
    return loaders, scaler
