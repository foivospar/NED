import sys
import os
sys.path.append(os.getcwd())
from manipulator.options.test_options import TestOptions
from manipulator.data.test_dataset import get_test_loader
from manipulator.models.model import create_model
from manipulator.checkpoint.checkpoint import CheckpointIO
from renderer.util.util import mkdirs
import torch
import numpy as np
import cv2
import os
from scipy.spatial.distance import cdist, euclidean
import pickle


# ported from "https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points"
def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1


@torch.no_grad()
def get_style_vectors(nets, opt, loader_src, loaders_ref):
    device = f'cuda:{opt.gpu_ids[0]}' if len(opt.gpu_ids) else 'cpu'

    s_refs = []

    if loaders_ref is not None:   # reference-guided
        # calculate average style vector of reference sequences
        for loader in loaders_ref:
            s_ref = []
            for x_ref, _ in loader:
                x_ref = x_ref.to(device)
                s_ref.append(nets.style_encoder(x_ref))
            s_ref = torch.cat(s_ref, dim=0)
            # geometric median
            s_ref = torch.from_numpy(geometric_median(s_ref.cpu().numpy())).type(torch.float32).to(device)
            s_ref = s_ref.view(1,-1)
            s_refs.append(s_ref)

    else:    # label-guided
        for e in opt.trg_emotions:
            if e not in opt.selected_emotions:
                print('Invalid target emotion!')
                exit(0)

        # generate style vectors
        z_trg = torch.randn(1, opt.latent_dim).to(device)
        for e in opt.trg_emotions:
            y = opt.selected_emotions.index(e)
            s_refs.append(nets.mapping_network(z_trg, torch.LongTensor(1).to(device).fill_(y)))

    return s_refs


if __name__ == '__main__':
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1
    opt.batch_size = 1
    device = f'cuda:{opt.gpu_ids[0]}' if len(opt.gpu_ids) else 'cpu'

    num_exp_coeffs = 51

    ### initialize dataset
    assert (opt.ref_dirs is None) != (opt.trg_emotions is None), 'Specify exactly one test mode'
    loader_src = get_test_loader(os.path.join(opt.celeb, 'DECA'), opt)
    loaders_ref = [get_test_loader(dir, opt) for dir in opt.ref_dirs] if opt.ref_dirs is not None else None

    ### initialize models
    nets = create_model(opt)

    ### load from checkpoint
    ckptio = CheckpointIO(os.path.join(opt.checkpoints_dir, '{:02d}_nets_finetuned.pth'), opt, len(opt.gpu_ids)>0, **nets)
    #ckptio = CheckpointIO(os.path.join(opt.checkpoints_dir, '{:02d}_nets.pth'), opt, len(opt.gpu_ids)>0, **nets)
    ckptio.load(opt.which_epoch)

    ### calculate style vector(s)
    s_refs = get_style_vectors(nets, opt, loader_src, loaders_ref)

    ### translate and store
    with torch.no_grad():
        gaussian = cv2.getGaussianKernel(opt.seq_len,-1)
        gaussian = torch.from_numpy(gaussian).float().to(device).repeat(1,num_exp_coeffs)
        output = torch.zeros(len(loader_src)+opt.seq_len-1, num_exp_coeffs).float().to(device)
        save_paths = []

        results_dir = os.path.join(opt.celeb, opt.exp_name, 'DECA')
        mkdirs(results_dir)
        for i, (x_src, paths) in enumerate(loader_src):

            # Prepare input sequence.
            x_src = x_src.to(device)

            # select style vector
            batch = len(output)/len(s_refs)
            s_ref = s_refs[int(i//batch)]
            # Translate sequence.
            x_fake = nets.generator(x_src, s_ref)

            output[i:i+opt.seq_len] = output[i:i+opt.seq_len] + torch.squeeze(x_fake, dim=0)*gaussian
            if i:
                save_paths.append(paths[-1][0])
            else:
                save_paths.extend([p[0] for p in paths])

        for i, (x_src, paths) in enumerate(loader_src):
            if i==0:
                # Prepare input sequence.
                x_src = x_src.to(device)
                # Translate sequence.
                x_fake = nets.generator(x_src, s_refs[0])
                output[:opt.seq_len-1] = torch.squeeze(x_fake, dim=0)[:-1]
            if i==len(loader_src)-1:
                # Prepare input sequence.
                x_src = x_src.to(device)
                # Translate sequence.
                x_fake = nets.generator(x_src, s_refs[-1])
                output[-opt.seq_len+1:] = torch.squeeze(x_fake, dim=0)[1:]


        # Save the translated sequence.
        output = output.cpu().numpy()
        for i in range(output.shape[0]):
            path = os.path.join(results_dir, save_paths[i])
            codedict = {}
            codedict['exp'] = output[i][1:].reshape((1,-1))
            pose = np.zeros((1,6), dtype=np.float32)
            pose[0,3] = output[i][0]
            codedict['pose'] = pose

            pickle.dump(codedict, open(path, 'wb'))
            print(f"Saving {path}")
