import argparse
import torch


def args_setting():
    hyper_para = argparse.ArgumentParser()
    hyper_para.add_argument('-pattern_dim', type=int, default=16, help='Number of Fluctuation Patterns')
    hyper_para.add_argument('-layer_num', type=int, default=2, help='Number of Attention Layers')
    hyper_para.add_argument('-slice_num', type=int, default=19, help='Number of Sliced Segments')
    self_args = hyper_para.parse_args()
    if torch.cuda.is_available():
        self_args.device = torch.device('cuda', 0)
    else:
        self_args.device = torch.device('cpu')
    return self_args


args = args_setting()
