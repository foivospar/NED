import os
import numpy as np
from munch import Munch
import torch
from torch.utils import data
import pickle


class MEAD(data.Dataset):
    """Dataset class for the MEAD dataset."""

    def __init__(self, opt, which='source', phase='train'):
        """Initialize the MEAD dataset."""

        self.which = which
        self.seq_len = opt.seq_len
        self.hop_len = opt.hop_len
        self.root = opt.train_root
        if phase == 'train':
            self.selected_actors = opt.selected_actors
        elif phase == 'val':
            self.selected_actors = opt.selected_actors_val

        self.selected_emotions = opt.selected_emotions

        self.seqs = []
        self.labels = []

        videos = []
        for actor in self.selected_actors:
            actor_root = os.path.join(self.root, '{}_deca.pkl'.format(actor))
            assert os.path.isfile(actor_root), '%s is not a valid file' % actor_root

            data_actor = pickle.load(open(actor_root, "rb"))
            videos.extend(data_actor)

        for v in videos:
            params, emotion = v
            params = np.concatenate((params[:,0:1], params[:,3:]),1)
            seqs = [params[x:x + self.seq_len] for x in
                                    range(0, params.shape[0], self.hop_len) if
                                    len(params[x:x + self.seq_len]) == self.seq_len]
            f = False
            for i, e in enumerate(self.selected_emotions):
                if e == emotion:
                    label = i
                    f = True
            if not f:
                print(emotion)
                # raise
            else:
                self.seqs.extend(seqs)
                self.labels.extend([label]*len(seqs))
        self.seqs = np.stack(self.seqs,axis=0)

        self.num_seqs = len(self.seqs)

        if self.which == 'reference':
            p = np.random.permutation(self.num_seqs)

            self.seqs = self.seqs[p]
            self.labels = np.array(self.labels)[p].tolist()

    def __getitem__(self, index):
        """Return one sequence and its corresponding label."""

        sequence = torch.FloatTensor(self.seqs[index])
        label = self.labels[index]

        return sequence, label

    def __len__(self):
        """Return the number of sequences."""
        return len(self.seqs)

def get_train_loader(opt, which):
    dataset = MEAD(opt, which)
    return data.DataLoader(dataset=dataset,
                           batch_size=opt.batch_size,
                           shuffle=True,
                           num_workers=opt.nThreads,
                           pin_memory=True)

def get_val_loader(opt, which):
    dataset = MEAD(opt, which, phase='val')
    return data.DataLoader(dataset=dataset,
                           batch_size=opt.batch_size,
                           shuffle=False,
                           num_workers=opt.nThreads,
                           pin_memory=True)


class InputFetcher:
    def __init__(self, loader, loader_ref, latent_dim=4):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_refs(self):
        try:
            x, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, y = next(self.iter_ref)
        return x, y

    def __next__(self):
        x, y = self._fetch_inputs()
        x_ref, y_ref = self._fetch_refs()
        z_trg = torch.randn(x.size(0), self.latent_dim)
        inputs = Munch(x_src=x, y_src=y, x_ref=x_ref, y_ref=y_ref, z_trg=z_trg)

        return inputs
