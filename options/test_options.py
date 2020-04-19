from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=400, help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='demo')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser

# 300 labeled 3000 unlabeled batch size 2 590s*200 = 118000s = 1966  min = 32.76h = 33 h
# 300 labeled 3000 unlabeled batch size 32 292s*200= 58400s  = 973   min = 16.21h = 16 h
# 300 labeled 1500 unlabeled batch size 2 292s*200 = 58400s  = 973   min = 16.21h = 16 h
# 300 labeled 1000 unlabeled batch size 2 195s*200 = 39000s  = 650   min = 10.83h = 11 h
# 300 labeled 600  unlabeled batch size 2 115s*200 = 33000s  = 550   min = 9.17 h = 9  h
# 100 labeled 1000 unlabeled batch size 2 195s *200= 39000s  = 650   min = 10.83h = 11 h
# 100 labeled 200  unlabeled batch size 2 42s *200 = 8400s   = 170   min = 2.83h  = 3  h

# 1: 20, 300 labeled 6000 unlabeled, batch size 2, 1195s*200= 239000s = 3988.8min = 66.48h = 66 h

# 1: 15, 300 labeled 4500 unlabeled, batch size 2, 895s*200 = 179000s = 2988.3min = 49.61h = 50 h

# 1: 10, 300 labeled 3000 unlabeled, batch size 2, 590s*200 = 118000s = 1966  min = 32.76h = 33 h

# 1: 5,  300 labeled 1500 unlabeled, batch size 2, 292s*200 =  58400s = 973   min = 16.21h = 16 h

# 1: 2,  300 labeled 600  unlabeled, batch size 2, 115s*200 =  33000s = 550   min =  9.17h = 9  h

# 1: 1,  300 labeled 300  unlabeled, batch size 2, 60s *200 =  12000s = 200   min =  3.33h = 3  h
