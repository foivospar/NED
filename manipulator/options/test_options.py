from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--which_epoch', type=int, default=2, help='which epoch to load?')
        self.parser.add_argument('--celeb', type=str, default='Pacino')
        self.parser.add_argument('--ref_dirs', type=str, nargs='+', help='Directories containing input reference sequences', default=None)
        self.parser.add_argument('--trg_emotions', type=str, nargs='+', choices=['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised', 'contempt'],
                                 help='Target emotions', default = None)
        self.parser.add_argument('--exp_name', type=str, help='Folder name to store the manipulated expression parameters', default = 'exp')
        self.isTrain = False
