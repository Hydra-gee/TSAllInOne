import PRNet

import argparse

import torch.cuda


# 变量名标识说明
# l:length 表示时间方向上的长度
# d:dimension 表示不同属性方向上的维度
def args_setting(cuda_id=0):
    hyper_para = argparse.ArgumentParser()
    # Basic Model Settings
    hyper_para.add_argument('-batch_size', type=int, default=64)
    hyper_para.add_argument('-epochs', type=int, default=30)
    hyper_para.add_argument('-learning_rate', type=float, default=1e-3)
    hyper_para.add_argument('-patience', type=int, default=5)
    # Dataset Settings
    hyper_para.add_argument('-dataset', type=str, default='Traffic')
    hyper_para.add_argument('-scale', type=int, default=4)  # l_history = l_pred * scale
    args = hyper_para.parse_args()
    dims = {'ETTh': 1, 'ETTm': 1, 'Exchange': 1, 'ECL': 1, 'Solar': 1, 'Weather':1, 'Stock':1, 'QPS':1, 'Traffic':1}
    lens = {'ETTh': 24, 'ETTm': 96, 'Exchange': 30, 'ECL': 96, 'Solar': 144, 'Weather':144, 'Stock':30, 'QPS':60, 'Traffic':24}
    args.l_pred = lens[args.dataset]
    args.d_in = dims[args.dataset]
    args.d_out = 1
    if torch.cuda.is_available():
        args.device = torch.device('cuda', cuda_id)
    else:
        args.device = torch.device('cpu')
    return args


args = args_setting(0)
process = PRNet.EXE(args)
process.count_parameter()
process.train()
process.test()