import csv
import datetime
import os
import time

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from argparse import Namespace

from data_loader import TSDataset
from algo.PRNet import PRNet
from algo.iTransformer import iTransformer
from algo.Crossformer import Crossformer
from algo.lightCTS import lightCTS

class Experiment:
    def __init__(self, expConfig, modelConfig) -> None:
        self.expConfig = expConfig
        self.modelConfig = modelConfig
        self.model_dict = {
            'PRNet': PRNet,
            'iTransformer': iTransformer,
            'Crossformer': Crossformer,
            'lightCTS':lightCTS
        }
        # self.model_config =
        print('Dataset:', expConfig['dataset'], '\tPrediction Length:', expConfig['pred_len'])
        self.model = self.model_dict[expConfig['model']].Model(expConfig, modelConfig[expConfig['model']]).to(expConfig['device'])
        # torch.set_float32_matmul_precision('high')
        # self.model = torch.compile(self.model)
        targetFolder = 'files/networks/' + expConfig['dataset']
        if not os.path.exists(targetFolder):
            os.makedirs(targetFolder)
        self.file_name = targetFolder + '/' + expConfig['model'] + '_' + str(expConfig['seq_len']) + '_' + str(
            expConfig['pred_len']) + '.pth'
        # if args.load == 'True':
        #     state_dict = torch.load(self.file_name)
        #     self.model.load_state_dict(state_dict)
        self.mse_func = torch.nn.MSELoss()
        self.mae_func = lambda x, y: torch.mean((torch.abs(x - y)))

    def _get_data(self, mode: str) -> DataLoader:
        dataset = TSDataset(self.expConfig['pred_len'], self.expConfig['seq_len'], self.expConfig['dim'],
                            self.expConfig['path'], self.expConfig['device'], mode)
        return DataLoader(dataset, batch_size=self.expConfig['batch_size'], shuffle=True)

    def _train_model(self, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        train_loss = 0
        for _, (x, y) in enumerate(loader):
            # currTime = time.time()
            optimizer.zero_grad()
            y_hat = self.model(x)
            loss = self.mse_func(y_hat, y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            # print('costTime:', time.time() - currTime)
        return train_loss / len(loader)

    def _eval_model(self, loader: DataLoader) -> (float, float):
        self.model.eval()
        mse_loss, mae_loss = 0, 0
        for _, (x, y) in enumerate(loader):
            y_hat = self.model(x)
            mse_loss += self.mse_func(y_hat, y).item()
            mae_loss += self.mae_func(y_hat, y).item()
        return mse_loss / len(loader), mae_loss / len(loader)

    def count_parameter(self) -> None:
        param_num = sum([param.nelement() for param in self.model.parameters()])
        print('Number of Parameters:', param_num)

    def train(self) -> None:
        train_loader = self._get_data('train')
        valid_loader = self._get_data('valid')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.expConfig['learning_rate'])
        patience, best_valid = 0, float('Inf')
        print('Total Epochs:', self.expConfig['epochs'])
        for epoch in range(self.expConfig['epochs']):
            currTime = time.time()
            train_loss = self._train_model(train_loader, optimizer)
            valid_loss, _ = self._eval_model(valid_loader)
            if valid_loss < best_valid:
                torch.save(self.model.state_dict(), self.file_name)
                best_valid = valid_loss
                patience = 0
            else:
                patience += 1
            print('Epoch', '%02d' % (epoch + 1), 'Train:', round(train_loss, 4), '\tValid:', round(valid_loss, 4),
                  '\tBest:', round(best_valid, 4), '\tPtc:', patience)
            print('costTime:', time.time() - currTime)
            if patience == self.expConfig['patience']:
                print('Early Stop!')
                break

    def test(self) -> None:
        state_dict = torch.load(self.file_name)
        self.model.load_state_dict(state_dict)
        test_loader = self._get_data('test')
        mse_loss, mae_loss = self._eval_model(test_loader)
        print(self.expConfig['dataset'], 'Test Loss')
        print('MSE: ', round(mse_loss, 4))
        print('MAE: ', round(mae_loss, 4))
        #写入日志
        row = [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               self.expConfig['dataset'],self.expConfig['model'], self.expConfig['seq_len'], self.expConfig['pred_len'],
               round(mse_loss, 4), round(mae_loss, 4)]
        with open("log.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
    # def visualize(self) -> None:
    #     dataset = TSDataset(self.args.pred_len, self.args.seq_len, self.args.dim, self.args.path, self.args.device, 'test')
    #     state_dict = torch.load(self.file_name)
    #     model = Model(self.args)
    #     model.load_state_dict(state_dict)
    #     model.eval()
    #     dim_id = int(input('Visual Dimension: '))
    #     index = int(input('Index: '))
    #     while index >= 0:
    #         if index < len(dataset):
    #             x, y = dataset[index]
    #             y_bar = model(x.unsqueeze(0))
    #             y_bar = y_bar.squeeze(0)[:, dim_id].detach().cpu().numpy()
    #             x, y = x[:, dim_id].detach().cpu().numpy(), y[:, dim_id].detach().cpu().numpy()
    #             plt.figure(figsize=(8, 2.4))
    #             plt.rcParams['font.sans-serif'] = ['Times New Roman']
    #             plt.plot(range(self.args.seq_len), x)
    #             plt.plot(range(self.args.seq_len, self.args.seq_len + self.args.pred_len), y, label='real')
    #             plt.plot(range(self.args.seq_len, self.args.seq_len + self.args.pred_len), y_bar, label='predict')
    #             plt.legend(fontsize=20)
    #             plt.yticks([])
    #             plt.tick_params(labelsize=20)
    #             plt.xlim([0, self.args.seq_len + self.args.pred_len])
    #             plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15)
    #             plt.savefig('files/figures/visual_' + self.args.dataset + '.pdf')
    #             plt.show()
    #         else:
    #             print('Out of Range!')
    #         index = int(input('Index: '))
