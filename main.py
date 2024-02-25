import json
import argparse
import torch.cuda

from model import PRNet
from data_loader import data_path_dict


def parse_args() -> argparse.Namespace:
    hyper_para = argparse.ArgumentParser()
    # Basic Settings
    hyper_para.add_argument('-cuda_id', type=int, default=0)
    hyper_para.add_argument('-batch_size', type=int, default=64)
    hyper_para.add_argument('-epochs', type=int, default=100)
    hyper_para.add_argument('-learning_rate', type=float, default=5e-4)
    hyper_para.add_argument('-patience', type=int, default=10, help='Early Stopping')
    hyper_para.add_argument('-hour_sampling', type=str, default='False')
    hyper_para.add_argument('-load', type=str, default='False')
    # Dataset Settings
    hyper_para.add_argument('-dataset', type=str, default='ETT', help='Dataset Name')
    hyper_para.add_argument('-pred_len', type=int, default=96, help='Prediction Length')
    # Model Settings
    hyper_para.add_argument('-layer_num', type=int, default=3, help='Number of Attention Layers')
    hyper_para.add_argument('-patch_num', type=int, default=19, help='Number of Segments')
    hyper_para.add_argument('-dropout', type=float, default=0.1, help='Dropout Probability')
    args = hyper_para.parse_args()

    with open('files/configs.json') as file:
        params = json.load(file)
        idx = 0 if args.hour_sampling == 'True' else 1
        args.path = data_path_dict[args.dataset][idx]
        args.patch_len = params[args.dataset]['patch_len'][idx]
        args.hidden_dim = params[args.dataset]['hidden_dim'][idx]
        args.dim = params[args.dataset]['dim']
    args.seq_len = args.patch_len * 4

    if torch.cuda.is_available():
        args.device = torch.device('cuda', args.cuda_id)
    else:
        args.device = torch.device('cpu')
    return args


if __name__ == '__main__':
    hyper_param = parse_args()
    model = PRNet(hyper_param)
    model.count_parameter()
    model.train()
    model.test()
    model.visualize()
