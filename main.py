import json
import argparse
import torch.cuda
import numpy as np
import random
from exp.experiment import Experiment
from data_loader import data_path_dict

fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

if __name__ == '__main__':
    # hyper_param = parse_args()
    with open('files/configs.json') as file:
        params = json.load(file)
        expConfig = params['expConfig']
        modelConfig = params['modelConfig']
        expConfig['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open('files/dataset_configs.json') as file2:
        datasetConfig = json.load(file2)
    for dataName in datasetConfig.keys():
        for pred in [1,2,3.5,7.5]:
            expConfig['patch_len'] = datasetConfig[dataName]['patch_len']
            expConfig['seq_len'] = int(4 * datasetConfig[dataName]['patch_len'])
            expConfig['pred_len'] = int(expConfig['seq_len'] * pred)
            expConfig['dataset'] = dataName
            expConfig['path'] = datasetConfig[expConfig['dataset']]['path']
            expConfig['dim'] = datasetConfig[expConfig['dataset']]['dim']

            exp = Experiment(expConfig,modelConfig)
            exp.count_parameter()
            exp.train()
            exp.test()
    # exp.visualize()
