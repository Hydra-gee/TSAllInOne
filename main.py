import argparse
import torch.cuda

from model import PRNet


hyper_para = argparse.ArgumentParser()
# Basic Settings
hyper_para.add_argument('-cuda_id', type=int, default=0)
hyper_para.add_argument('-batch_size', type=int, default=64)
hyper_para.add_argument('-epochs', type=int, default=100)
hyper_para.add_argument('-learning_rate', type=float, default=1e-3)
hyper_para.add_argument('-patience', type=int, default=10, help='Early Stopping')
# Dataset Settings
hyper_para.add_argument('-dataset', type=str, default='Weather')
hyper_para.add_argument('-scale', type=int, default=4, help='seq_len = pred_len * scale')
hyper_para.add_argument('-pred_scale', type=float, default=1)
channel_dims = {'ECL': 370, 'ETTh': 7, 'ETTm': 7, 'Exchange': 8, 'QPS': 10, 'Solar': 137, 'Traffic': 862, 'Weather': 1}  # dimensions of datasets
pred_lens = {'ECL': 96, 'ETTh': 24, 'ETTm': 96, 'Exchange': 30, 'QPS': 60, 'Solar': 144, 'Traffic': 24, 'Weather': 144}  # forecasting lengths of datasets
# Model Settings
hyper_para.add_argument('-layer_num', type=int, default=2, help='Number of Attention Layers')
hyper_para.add_argument('-seg_num', type=int, default=19, help='Number of Sliced Segments')
hyper_para.add_argument('-dropout', type=float, default=0.1, help='Dropout Probability')
args = hyper_para.parse_args()

args.pred_len = int(pred_lens[args.dataset] * args.pred_scale)
args.channel_dim = channel_dims[args.dataset]

if args.dataset in ['ETTh', 'ETTm']:
    args.spatial = True
else:
    args.spatial = False

if torch.cuda.is_available():
    args.device = torch.device('cuda', args.cuda_id)
else:
    args.device = torch.device('cpu')


if __name__ == '__main__':
    model = PRNet(args)
    model.count_parameter()
    model.train()
    model.test()
