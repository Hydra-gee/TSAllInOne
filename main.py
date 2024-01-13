import json
import argparse
import torch.cuda

from model import PRNet


def parse_args():
    hyper_para = argparse.ArgumentParser()
    # Basic Settings
    hyper_para.add_argument('-cuda_id', type=int, default=0)
    hyper_para.add_argument('-batch_size', type=int, default=64)
    hyper_para.add_argument('-epochs', type=int, default=100)
    hyper_para.add_argument('-learning_rate', type=float, default=5e-4)
    hyper_para.add_argument('-patience', type=int, default=10, help='Early Stopping')
    hyper_para.add_argument('-load', type=str, default='False')
    # Dataset Settings
    hyper_para.add_argument('-dataset', type=str, default='Weather_h', help='Dataset Name')
    hyper_para.add_argument('-pred_len', type=int, default=24, help='Prediction Length')
    # Model Settings
    hyper_para.add_argument('-layer_num', type=int, default=3, help='Number of Attention Layers')
    hyper_para.add_argument('-patch_num', type=int, default=19, help='Number of Segments')
    hyper_para.add_argument('-dropout', type=float, default=0.1, help='Dropout Probability')
    args = hyper_para.parse_args()

    with open('files/configs.json') as file:
        params = json.load(file)
        args.patch_len = params[args.dataset]['patch_len']
        args.embed_dim = params[args.dataset]['embed_dim']
        args.dim = params[args.dataset]['dim']
    args.seq_len = args.patch_len * 4

    if torch.cuda.is_available():
        args.device = torch.device('cuda', args.cuda_id)
    else:
        args.device = torch.device('cpu')
    return args


if __name__ == '__main__':
    hyper_params = parse_args()
    model = PRNet(hyper_params)
    model.count_parameter()
    model.train()
    model.test()
    model.visualize()
