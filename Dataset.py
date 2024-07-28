import torch
from torch.utils.data import Dataset
class Dataseting(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float() # convert array into tensor
        self.target = torch.from_numpy(target).long() # convert array into tensor
        self.transform = transform # applies transformers
        
    def __getitem__(self, index):
        data = self.data[index]
        target = self.target[index]
        
        if self.transform:
            data = self.transform(data)
        
        return data, target
    
    def __len__(self):
        return len(self.data)