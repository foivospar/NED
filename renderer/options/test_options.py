from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--time_fwd_pass', action='store_true', help='Show the forward pass time for synthesizing each frame.')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--exp_name', type=str, default='', help='Subfolder for specific experiment (empty string for self reenactment)')
        self.parser.add_argument('--self_name', type=str, default='Folder name to store self-reenactment results', help='')
        self.isTrain = False
