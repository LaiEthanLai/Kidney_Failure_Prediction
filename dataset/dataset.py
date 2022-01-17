import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class Kidney_Dataset(Dataset):
    def __init__(self, data, label, transform=None):
        self.transform = transform
        self.data = pd.read_csv(data, header=None, dtype="float64").values
        self.label = pd.read_csv(label, header=None, dtype="float64").values

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = {"Data": torch.from_numpy(self.data[idx]), "Label": torch.from_numpy(self.label[idx]).long()}
        
        return sample