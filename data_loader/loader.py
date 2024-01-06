import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class TimeSeries(Dataset):
    def __init__(self, pred_len, seq_len):
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.data = None

    def __getitem__(self, item):
        x_begin = item
        x_end = x_begin + self.seq_len
        y_begin = x_end
        y_end = x_end + self.pred_len
        x = self.data[x_begin:x_end]
        y = self.data[y_begin:y_end]
        return x, y

    def __len__(self):
        return self.data.shape[0] - self.seq_len - self.pred_len + 1

    def _normalize(self):
        avg = torch.mean(self.data, dim=0, keepdim=True)
        std = torch.std(self.data, dim=0, keepdim=True)
        self.data = (self.data - avg) / std

    def _split(self, flag):
        if flag == 'train':
            self.data = self.data[:int(self.data.shape[0] * 0.7)]
        elif flag == 'valid':
            self.data = self.data[int(self.data.shape[0] * 0.7):int(self.data.shape[0] * 0.8)]
        elif flag == 'test':
            self.data = self.data[int(self.data.shape[0] * 0.8):]


class ECL(TimeSeries):
    def __init__(self, device, pred_len, seq_len, channel_dim, flag='train'):
        super().__init__(pred_len, seq_len)
        dataset = pd.read_csv('dataset/LD2011_2014.txt', delimiter=';')
        assert channel_dim < dataset.shape[1]
        self.data = torch.tensor(dataset.iloc[:, -channel_dim:].values, device=device, dtype=torch.float32)
        self._normalize()
        self._split(flag)


class ETTh(TimeSeries):
    def __init__(self, device, pred_len, seq_len, channel_dim, flag='train', idx=1):
        super().__init__(pred_len, seq_len)
        dataset = pd.read_csv('dataset/ETT/ETTh' + str(idx) + '.csv')
        assert channel_dim < dataset.shape[1]
        self.data = torch.tensor(dataset.iloc[:, -channel_dim:].values, device=device, dtype=torch.float32)
        self._normalize()
        self._split(flag)


class ETTm(TimeSeries):
    def __init__(self, device, pred_len, seq_len, channel_dim, flag='train', idx=1):
        super().__init__(pred_len, seq_len)
        dataset = pd.read_csv('dataset/ETT/ETTm' + str(idx) + '.csv')
        assert channel_dim < dataset.shape[1]
        self.data = torch.tensor(dataset.iloc[:, -channel_dim:].values, device=device, dtype=torch.float32)
        self._normalize()
        self._split(flag)


class Exchange(TimeSeries):
    def __init__(self, device, pred_len, seq_len, channel_dim, flag='train'):
        super().__init__(pred_len, seq_len)
        dataset = np.loadtxt('dataset/exchange_rate.csv', delimiter=',')
        assert channel_dim <= dataset.shape[1]
        self.data = torch.tensor(dataset[:, -channel_dim:], device=device, dtype=torch.float32)
        self._normalize()
        self._split(flag)


class Solar(TimeSeries):
    def __init__(self, device, pred_len, seq_len, channel_dim, flag='train'):
        super().__init__(pred_len, seq_len)
        dataset = np.loadtxt('dataset/solar_AL.csv', delimiter=',')
        assert channel_dim <= dataset.shape[1]
        self.data = torch.tensor(dataset[:, -channel_dim:], device=device, dtype=torch.float32)
        self._normalize()
        self._split(flag)


class Traffic(TimeSeries):
    def __init__(self, device, pred_len, seq_len, channel_dim, flag='train'):
        super().__init__(pred_len, seq_len)
        dataset = np.loadtxt('dataset/traffic.csv', delimiter=',')
        assert channel_dim <= dataset.shape[1]
        self.data = torch.tensor(dataset[:, -channel_dim:], device=device, dtype=torch.float32)
        self._normalize()
        self._split(flag)


class Weather(TimeSeries):
    def __init__(self, device, pred_len, seq_len, channel_dim, flag='train'):
        super().__init__(pred_len, seq_len)
        dataset = pd.read_csv('dataset/weather.csv', encoding='ISO-8859-1')
        assert channel_dim == 1
        self.data = torch.tensor(dataset.iloc[:, -1:].values, device=device, dtype=torch.float32)
        self._normalize()
        self._split(flag)


class QPS(TimeSeries):
    def __init__(self, device, pred_len, seq_len, channel_dim, flag='train'):
        super().__init__(pred_len, seq_len)
        dataset = pd.read_csv('dataset/QPS.csv')
        assert channel_dim < dataset.shape[1]
        self.data = torch.tensor(dataset.iloc[:, 1:].values, device=device, dtype=torch.float32)
        self._normalize()
        self._split(flag)
