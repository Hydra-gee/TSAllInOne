import argparse
import torch.cuda

import model


# 变量名标识说明
# l:length 表示时间方向上的长度
# d:dimension 表示不同属性方向上的维度
hyper_para = argparse.ArgumentParser()
# Basic Settings
hyper_para.add_argument('-cuda_id', type=int, default=0)
hyper_para.add_argument('-batch_size', type=int, default=64)
hyper_para.add_argument('-epochs', type=int, default=30)
hyper_para.add_argument('-learning_rate', type=float, default=1e-3)
hyper_para.add_argument('-patience', type=int, default=5, help='Early Stopping')
# Dataset Settings
hyper_para.add_argument('-dataset', type=str, default='ETTm')
hyper_para.add_argument('-scale', type=int, default=4, help='l_history = l_pred * scale')
dims = {'ETTh': 7, 'ETTm': 7, 'Exchange': 8, 'ECL': 1,'Traffic':1, 'Solar': 1, 'Weather':1, 'QPS':1}
lens = {'ETTh': 24, 'ETTm': 96, 'Exchange': 30, 'ECL': 96, 'Solar': 144, 'Weather':144, 'QPS':60, 'Traffic':144}
# Model Settings
hyper_para.add_argument('-n_pattern', type=int, default=16, help='Number of Fluctuation Patterns')
hyper_para.add_argument('-layer_num', type=int, default=2, help='Number of Attention Layers')
hyper_para.add_argument('-segment_num', type=int, default=19, help='Number of Sliced Segments')
args = hyper_para.parse_args()
args.l_pred = lens[args.dataset]
args.d_in = dims[args.dataset]
args.d_out = 1
if torch.cuda.is_available():
    args.device = torch.device('cuda', args.cuda_id)
else:
    args.device = torch.device('cpu')


process = model.EXE(args)
process.count_parameter()
process.train()
process.test()