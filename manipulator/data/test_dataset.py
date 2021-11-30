import os
import numpy as np

import torch
from torch.utils import data
import pickle

FILE_EXTENSIONS = ['.txt', '.pkl']

def is_exp_file(filename):
    return any(filename.endswith(extension) for extension in FILE_EXTENSIONS)

class test(data.Dataset):
    """Dataset class for test dataset."""

    def __init__(self, dataroot, opt):
        """Initialize the dataset."""

        self.root = dataroot
        paths = [os.path.join(self.root, f) for f in sorted(os.listdir(self.root)) if is_exp_file(f)]
        self.seq_paths = [paths[x:x+opt.seq_len] for x in range(0, len(paths), 1) if len(paths[x:x+opt.seq_len])==opt.seq_len]

    def __getitem__(self, index):
        """Return one sequence and its corresponding paths."""

        paths = self.seq_paths[index]
        params = []
        for pth in paths:
            deca = pickle.load(open(pth,'rb'))
            params.append(np.concatenate((deca['pose'][:,3:4], deca['exp']), 1))
        sequence = torch.FloatTensor(np.concatenate(params, 0))

        return sequence, [os.path.basename(p) for p in paths]


    def __len__(self):
        """Return the number of sequences."""
        return len(self.seq_paths)

def get_test_loader(root, opt):
    if root is None:
        return None
    else:
        dataset = test(root, opt)
        return data.DataLoader(dataset=dataset,
                               batch_size=opt.batch_size,
                               shuffle=False,
                               num_workers=opt.nThreads,
                               pin_memory=True)
