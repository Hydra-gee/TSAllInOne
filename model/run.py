import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader import *
from model.structure import Model


class PRNet:
    def __init__(self, args, load=False):
        self.args = args
        self.dict = {'ECL': ECL, 'ETTh': ETTh, 'ETTm': ETTm, 'Exchange': Exchange, 'QPS': QPS, 'Solar': Solar, 'Traffic': Traffic, 'Weather': Weather}
        print('Dataset:', args.dataset)
        print('Prediction Length:', args.pred_len)
        self.seq_len = args.pred_len * args.scale
        self.best_valid = np.Inf
        self.best_epoch = 0
        if load:
            self.model = torch.load('files/networks/' + args.dataset + '_' + str(args.pred_len) + '.pth').to(args.device)
        else:
            self.model = Model(args).to(args.device)
        self.mse_func = torch.nn.MSELoss()
        self.mae_func = lambda x, y: torch.mean((torch.abs(x-y)))

    def _get_data(self, flag='train'):
        dataset = self.dict[self.args.dataset](self.args.device, self.args.pred_len, self.seq_len, self.args.channel_dim, flag)
        dataset = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
        return dataset

    def train(self):
        print('Total Epochs:', self.args.epochs)
        train_loader = self._get_data('train')
        valid_loader = self._get_data('valid')
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        patient_epoch = 0
        for epoch in range(self.args.epochs):
            print('Epoch', epoch + 1)
            # training
            self.model.train()
            start_time = time.time()
            batch_num, train_loss = 0, 0
            for _, (x, y) in enumerate(train_loader):
                avg = torch.mean(x, dim=1, keepdim=True)
                x = x - avg
                y = y - avg
                optimizer.zero_grad()
                season, trend = self.model(x)
                loss = self.mse_func(season + trend, y)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                batch_num += 1
            end_time = time.time()
            print('Training Time', round(end_time - start_time, 2), 's')
            print('Training Loss (MSE): ', round(train_loss / batch_num, 4))
            # validation
            self.model.eval()
            batch_num, valid_loss = 0, 0
            for _, (x, y) in enumerate(valid_loader):
                avg = torch.mean(x, dim=1, keepdim=True)
                x = x - avg
                season, trend = self.model(x)
                loss = self.mse_func(season + trend + avg, y)
                valid_loss += loss.item()
                batch_num += 1
            print('Validation Loss (MSE): ', round(valid_loss / batch_num, 4))
            if valid_loss < self.best_valid:
                torch.save(self.model, 'files/networks/' + self.args.dataset + '_' + str(self.args.pred_len) + '.pth')
                self.best_valid = valid_loss
                patient_epoch = 0
            else:
                patient_epoch += 1
            if patient_epoch > self.args.patience:
                print('Early Stop!')
                break

    def test(self):
        self.model = torch.load('files/networks/' + self.args.dataset + '_' + str(self.args.pred_len) + '.pth').to(self.args.device)
        test_loader = self._get_data('test')
        batch_num, mse_loss, mae_loss = 0, 0, 0
        self.model.eval()
        for i, (x, y) in enumerate(test_loader):
            avg = torch.mean(x, dim=1, keepdim=True)
            x = x - avg
            season, trend = self.model(x)
            output = season + trend + avg
            mse_loss += self.mse_func(output, y).item()
            mae_loss += self.mae_func(output, y).item()
            batch_num += 1
        print('Test Loss')
        print('MSE: ', round(mse_loss / batch_num, 4))
        print('MAE: ', round(mae_loss / batch_num, 4))

    def count_parameter(self):
        total = sum([param.nelement() for param in self.model.parameters()])
        print('Number of Parameters:', total)
