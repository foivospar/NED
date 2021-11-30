import argparse
import os
from renderer.util.util import mkdirs
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # model arguments
        self.parser.add_argument('--seq_len', type=int, default=10, help='Length of exp. coeffs. sequence')
        self.parser.add_argument('--hop_len', type=int, default=1, help='Hop Length (set to 1 by default for test)')
        self.parser.add_argument('--selected_emotions', type=str, nargs='+', help='Subset (or all) of the 8 basic emotions',
                                 default=['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised'],
                                 choices=['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised', 'contempt'])

        self.parser.add_argument('--batch_size', type=int, default=64, help='input batch size')

        # network arch
        self.parser.add_argument('--latent_dim', type=int, default=4, help='Latent vector dimension')
        self.parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension of mapping network')
        self.parser.add_argument('--style_dim', type=int, default=16, help='Style code dimension')

        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./manipulator_checkpoints', help='models are saved here')

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        mkdirs(self.opt.checkpoints_dir)
        if save:
            file_name = os.path.join(self.opt.checkpoints_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
