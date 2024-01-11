import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from data_loader import data_dict
from model.model import Model


class PRNet:
    def __init__(self, args):
        self.args = args
        print('Dataset:', args.dataset, '\tPrediction Length:', args.pred_len)
        self.model = Model(args).to(args.device)
        if args.load == 'True':
            state_dict = torch.load('files/networks/' + args.dataset + '_' + str(args.pred_len) + '.pth')
            self.model.load_state_dict(state_dict)
        self.mse_func = torch.nn.MSELoss()
        self.mae_func = lambda x, y: torch.mean((torch.abs(x - y)))

    def _get_data(self, mode):
        dataset = data_dict[self.args.dataset](self.args.device, self.args.pred_len, self.args.seq_len, self.args.dim, mode)
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

    def count_parameter(self):
        total = sum([param.nelement() for param in self.model.parameters()])
        print('Number of Parameters:', total)

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
                torch.save(self.model.state_dict(), 'files/networks/' + self.args.dataset + '_' + str(self.args.pred_len) + '.pth')
                best_valid = valid_loss
                patience = 0
            else:
                patience += 1
            print('Epoch', '%02d' % (epoch + 1), 'Train:', round(train_loss, 4), '\tValid:', round(valid_loss, 4), patience, round(best_valid, 4))
            if patience == self.args.patience:
                print('Early Stop!')
                break

    def test(self):
        state_dict = torch.load('files/networks/' + self.args.dataset + '_' + str(self.args.pred_len) + '.pth')
        self.model.load_state_dict(state_dict)
        test_loader = self._get_data('test')
        mse_loss, mae_loss = self._eval_model(test_loader)
        print(self.args.dataset, 'Test Loss')
        print('MSE: ', round(mse_loss, 4))
        print('MAE: ', round(mae_loss, 4))

    def visualize(self):
        dataset = data_dict[self.args.dataset](self.args.device, self.args.pred_len, self.args.seq_len, self.args.dim, 'test')
        state_dict = torch.load('files/networks/' + self.args.dataset + '_' + str(self.args.pred_len) + '.pth')
        self.model.load_state_dict(state_dict)
        self.model.eval()
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        dim = int(input('Visual Dimension: '))
        index = int(input('Index:'))
        while index >= 0:
            if index < len(dataset):
                x, y = dataset[index]
                season, trend = self.model(x.unsqueeze(0))
                y_bar = (season + trend).squeeze(0)[:, dim].detach().cpu().numpy()
                x, y = x[:, dim].detach().cpu().numpy(), y[:, dim].detach().cpu().numpy()
                plt.figure(figsize=(8, 2.4))
                plt.plot(range(self.args.seq_len), x)
                plt.plot(range(self.args.seq_len, self.args.seq_len + self.args.pred_len), y, label='real')
                plt.plot(range(self.args.seq_len, self.args.seq_len + self.args.pred_len), y_bar, label='predict')
                plt.legend(fontsize=20)
                plt.yticks([])
                plt.tick_params(labelsize=20)
                plt.xlim([0, self.args.seq_len + self.args.pred_len])
                plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15)
                plt.savefig('files/figures/visual_' + self.args.dataset + '.pdf')
                plt.show()
            else:
                print('Out of Range!')
            index = int(input('Index:'))
