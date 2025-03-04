import random
import torch
import numpy as np
from torch.utils.data import Dataset


class SpectraDataset(Dataset):
    def __init__(self, data_split, seed=42):
        self.data_split = data_split
        assert data_split in ['train','test'], f"Invalid data_split: {self.data_split}"

        self.Rdataset = torch.load("./NeuralAdataset/R_dataset_1269x31.pt", weights_only=True)  # (1269, 31)
        self.Idataset = torch.load("./NeuralAdataset/I_dataset_449x31.pt", weights_only=True)  # (449, 31)

        set_random_seed(seed)

        train_ratio = 0.8
        num_R_train = int(len(self.Rdataset) * train_ratio)
        num_I_train = int(len(self.Idataset) * train_ratio)

        if self.data_split == 'train':
            self.Rdataset = self.Rdataset[:num_R_train]
            self.Idataset = self.Idataset[:num_I_train]
        else:
            self.Rdataset = self.Rdataset[num_R_train:]
            self.Idataset = self.Idataset[num_I_train:]

    def __len__(self):
        return 500000   # control the number of samples

    def __getitem__(self, idx):
        idx_R1, idx_R2, idx_R3, idx_R4 = random.sample(range(len(self.Rdataset)), 4)
        R1 = self.Rdataset[idx_R1] + self.Rdataset[idx_R2]
        R2 = self.Rdataset[idx_R3] + self.Rdataset[idx_R4]

        idx_L1, idx_L2, idx_L3, idx_L4 = random.sample(range(len(self.Rdataset)), 4)
        L1 = self.Rdataset[idx_L1] + self.Rdataset[idx_L2]
        L2 = self.Rdataset[idx_L3] + self.Rdataset[idx_L4]

        target = (R1 * L1) + (R2 * L2)
        
        return R1, L1, R2, L2, target
    

def set_random_seed(seed):
    random.seed(seed)  # Python 内置随机数
    np.random.seed(seed)  # NumPy 随机数
    torch.manual_seed(seed)  # PyTorch CPU 随机数
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU 随机数（多 GPU）
    torch.backends.cudnn.deterministic = True  # 确保卷积操作可复现
    torch.backends.cudnn.benchmark = False  # 关闭自动优化（可选）