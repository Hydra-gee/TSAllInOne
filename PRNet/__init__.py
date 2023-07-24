import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from Loader import ETT_Minute, ETT_Hour, Electricity, Exchange, Solar, Weather, Stock, QPS, Traffic
from PRNet.structure import Model, PeriodNet, TrendNet
from PRNet.tool import decomposition


class EXE:
    def __init__(self, args, load=False):
        self.args = args
        self.dict = {'ETTm': ETT_Minute, 'ETTh': ETT_Hour, 'ECL': Electricity, 'Exchange': Exchange, 'Traffic':Traffic, 'Solar': Solar, 'Weather': Weather, 'Stock':Stock, 'QPS':QPS}
        print('Dataset:', args.dataset)
        print('Prediction Length:', args.l_pred)
        self.data = self.dict[args.dataset]
        self.l_seq = args.l_pred * args.scale
        self.best_valid = np.Inf
        self.best_epoch = 0
        if load:
            self.model = torch.load('Model/' + args.dataset + '_' + str(self.args.l_pred) + '.pth').to(args.device)
        else:
            self.model = Model(args).to(args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-5)
        self.criterion = torch.nn.MSELoss()
        self.mae_fun = lambda x, y: torch.mean((torch.abs(x-y)))

    def get_data(self, flag='train'):
        dataset = self.data(self.args.device, self.args.l_pred, self.l_seq, self.args.d_in, flag)
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
        return loader

    def batch_process(self, x, y):
        res, avg = self.model(x)
        r_y, a_y = decomposition(y, self.args.l_pred)
        return res, avg, r_y[:, :, -self.args.d_out:], a_y[:, :, -self.args.d_out:]

    def train(self):
        print('Total Epochs:', self.args.epochs)
        train_loader = self.get_data('train')
        valid_loader = self.get_data('valid')
        patient_epoch = 0
        for epoch in range(self.args.epochs):
            print('Epoch', epoch + 1)
            # training
            self.model.train()
            start_time = time.time()
            iter_count, train_loss = 0, 0
            for i, (x, y) in enumerate(train_loader):
                iter_count += 1
                self.optimizer.zero_grad()
                res, avg, r_y, a_y = self.batch_process(x, y)
                loss = self.criterion(res, r_y) + self.criterion(avg, a_y)
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            end_time = time.time()
            print('Training Time', round(end_time - start_time, 2))
            print('Training Loss (MSE): ', round(train_loss / iter_count, 4))
            # validation
            self.model.eval()
            iter_count, valid_loss = 0, 0
            for i, (x, y) in enumerate(valid_loader):
                iter_count += 1
                res, avg, r_y, a_y = self.batch_process(x, y)
                output = res + avg
                label = r_y + a_y
                loss = self.criterion(output, label)
                valid_loss += loss.item()
            print('Validation Loss (MSE): ', round(valid_loss / iter_count, 4))
            if valid_loss < self.best_valid:
                torch.save(self.model, 'Model/' + self.args.dataset + '_' + str(self.args.l_pred) + '.pth')
                self.best_valid = valid_loss
                patient_epoch = 0
            else:
                patient_epoch += 1
            if patient_epoch >= self.args.patience:
                print('Early Stop!')
                break

    def test(self):
        self.model = torch.load('Model/' + self.args.dataset + '_' + str(self.args.l_pred) + '.pth').to(self.args.device)
        test_loader = self.get_data('test')
        iter_count, mse_loss, mae_loss = 0, 0, 0
        self.model.eval()
        for i, (x, y) in enumerate(test_loader):
            iter_count += 1
            res, avg, r_y, a_y = self.batch_process(x, y)
            output = (res + avg)
            label = r_y + a_y
            mse_loss += self.criterion(output, label).item()
            mae_loss += self.mae_fun(output, label).item()
        print('Test Loss')
        print('MSE: ', round(mse_loss / iter_count, 4))
        print('MAE: ', round(mae_loss / iter_count, 4))

    def count_parameter(self):
        total = sum([param.nelement() for param in self.model.parameters()])
        print('Number of Parameters:', total)