import torch
from torch.utils.data import DataLoader

from data_loader import *
from model.model import Model


class PRNet:
    def __init__(self, args):
        self.args = args
        self.dict = {
            'Electricity': Electricity,
            'ETTh': ETTh, 'ETTm': ETTm,
            'Exchange': Exchange,
            'QPS': QPS,
            'Solar': Solar,
            'Traffic': Traffic,
            'Weather': Weather
        }
        print('Dataset:', args.dataset, '\tPrediction Length:', args.pred_len)
        if args.load == 'True':
            self.model = torch.load('files/networks/' + args.dataset + '_' + str(args.pred_len) + '.pth').to(args.device)
        else:
            self.model = Model(args).to(args.device)
        self.mse_func = torch.nn.MSELoss()
        self.mae_func = lambda x, y: torch.mean((torch.abs(x - y)))

    def _get_data(self, mode):
        dataset = self.dict[self.args.dataset](self.args.device, self.args.pred_len, self.args.seq_len, self.args.dim, mode)
        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

    def _train_model(self, loader, optimizer):
        self.model.train()
        train_loss = 0
        for _, (x, y) in enumerate(loader):
            optimizer.zero_grad()
            season, trend = self.model(x)
            loss = self.mse_func(season + trend, y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        return train_loss / len(loader)

    def _eval_model(self, loader):
        self.model.eval()
        mse_loss, mae_loss = 0, 0
        for _, (x, y) in enumerate(loader):
            season, trend = self.model(x)
            mse_loss += self.mse_func(season + trend, y).item()
            mae_loss += self.mae_func(season + trend, y).item()
        return mse_loss / len(loader), mae_loss / len(loader)

    def train(self):
        print('Total Epochs:', self.args.epochs)
        train_loader = self._get_data('train')
        valid_loader = self._get_data('valid')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        patience, best_valid = 0, float('Inf')
        for epoch in range(self.args.epochs):
            train_loss = self._train_model(train_loader, optimizer)
            valid_loss, _ = self._eval_model(valid_loader)
            if valid_loss < best_valid:
                torch.save(self.model, 'files/networks/' + self.args.dataset + '_' + str(self.args.pred_len) + '.pth')
                best_valid = valid_loss
                patience = 0
            else:
                patience += 1
            print('Epoch', '%02d' % (epoch + 1), 'Train:', round(train_loss, 4), '\tValid:', round(valid_loss, 4), patience, round(best_valid, 4))
            if patience == self.args.patience:
                print('Early Stop!')
                break

    def test(self):
        self.model = torch.load('files/networks/' + self.args.dataset + '_' + str(self.args.pred_len) + '.pth').to(self.args.device)
        test_loader = self._get_data('test')
        mse_loss, mae_loss = self._eval_model(test_loader)
        print(self.args.dataset, 'Test Loss')
        print('MSE: ', round(mse_loss, 4))
        print('MAE: ', round(mae_loss, 4))

    def count_parameter(self):
        total = sum([param.nelement() for param in self.model.parameters()])
        print('Number of Parameters:', total)
