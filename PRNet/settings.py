import argparse
import torch


def args_setting():
    hyper_para = argparse.ArgumentParser()
    layer_num = 2
    seg_num = 13
    hyper_para.add_argument('-pattern_dim', type=int, default=16, help='Number of Fluctuation Patterns')
    self_args = hyper_para.parse_args()
    self_args.period_layer = layer_num
    self_args.trend_layer = layer_num
    self_args.slice_num = seg_num
    self_args.embed_dim = seg_num
    if torch.cuda.is_available():
        self_args.device = torch.device('cuda', 0)
    else:
        self_args.device = torch.device('cpu')
    return self_args


args = args_setting()
