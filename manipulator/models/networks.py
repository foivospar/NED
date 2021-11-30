import torch
import torch.nn as nn
import torch.nn.functional as F

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=4, hidden_dim = 64, style_dim=16, n_emotions = 7):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, hidden_dim)]
        layers += [nn.ReLU()]
        for _ in range(2):
            layers += [nn.Linear(hidden_dim, hidden_dim)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(n_emotions):
            self.unshared += [nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, n_emotions, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s

class Generator(nn.Module):

    def __init__(self, style_dim = 16, num_exp_coeffs = 51):

        super(Generator, self).__init__()
        self.seq1 = nn.Sequential(nn.Linear(num_exp_coeffs+style_dim,512), nn.ReLU(), nn.Linear(512,512), nn.ReLU())
        self.LSTM = nn.LSTM(512, 512, 2, batch_first=True, bidirectional=True)
        self.seq2 = nn.Sequential(nn.Linear(1024,1024), nn.ReLU(), nn.Linear(1024,1024), nn.ReLU(),
                                  nn.Linear(1024,num_exp_coeffs))

    def forward(self, x, s):

        s = s.view(s.size(0), 1, s.size(1))
        s = s.repeat(1, x.size(1), 1)
        x = torch.cat([x, s], dim=2)

        self.LSTM.flatten_parameters()
        out, _ = self.LSTM(self.seq1(x))
        out = self.seq2(out)
        
        return out

class Discriminator(nn.Module):

    def __init__(self, n_emotions = 7, num_exp_coeffs=28):

        super(Discriminator, self).__init__()

        self.seq1 = nn.Sequential(nn.Linear(num_exp_coeffs,512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.LSTM = nn.LSTM(256, 128, 2, batch_first=True, bidirectional = True)
        self.seq2 = nn.Sequential(nn.Linear(256,128), nn.ReLU(), nn.Linear(128,64), nn.ReLU(), nn.Linear(64, n_emotions))

    def forward(self, x, y):

        self.LSTM.flatten_parameters()
        out, _ = self.LSTM(self.seq1(x))
        out = torch.mean(self.seq2(out),1)  # (batch, n_emotions)

        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        return out

class StyleEncoder(nn.Module):
    def __init__(self, style_dim=16, num_exp_coeffs=28):
        super().__init__()

        self.seq1 = nn.Sequential(nn.Linear(num_exp_coeffs,512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.LSTM = nn.LSTM(256, 128, 2, batch_first=True, bidirectional = True)
        self.seq2 = nn.Sequential(nn.Linear(256,128), nn.ReLU(), nn.Linear(128,64), nn.ReLU(), nn.Linear(64, style_dim))

    def forward(self, x):

        self.LSTM.flatten_parameters()
        h, _ = self.LSTM(self.seq1(x))
        s = self.seq2(h)  # (batch, seq_len, style_dim)
        return torch.mean(s, 1)   # (batch, style_dim)
