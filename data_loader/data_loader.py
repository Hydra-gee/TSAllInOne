import torch
from torch.utils.data import Dataset
import pandas as pd


class TimeSeries(Dataset):
    def __init__(self, pred_len, seq_len):
        self.pred_len = pred_len
        self.seq_len = seq_len

    def _normalize(self):
        avg = torch.mean(self.data, dim=0, keepdim=True)
        std = torch.std(self.data, dim=0, keepdim=True)
        self.data = (self.data - avg) / std

    def _split(self, mode):
        total_len = self.data.shape[0] - self.seq_len - self.pred_len * 3 + 3
        max_train_idx = int(total_len * 0.7) + self.seq_len + self.pred_len - 1
        max_valid_idx = int(total_len * 0.8) + self.seq_len + self.pred_len * 2 - 2
        if mode == 'train':
            self.data = self.data[:max_train_idx]
        elif mode == 'valid':
            self.data = self.data[max_train_idx + self.pred_len - 1: max_valid_idx]
        elif mode == 'test':
            self.data = self.data[max_train_idx + self.pred_len - 1:]

    def __len__(self):
        return self.data.shape[0] - self.seq_len - self.pred_len + 1

    def __getitem__(self, item):
        left, mid, right = item, item + self.seq_len, item + self.seq_len + self.pred_len
        x = self.data[left: mid]
        y = self.data[mid: right]
        return x, y


# Electricity Consumption
class Electricity(TimeSeries):
    def __init__(self, pred_len, seq_len, dim, mode):
        super().__init__(pred_len, seq_len)
        dataset = pd.read_csv('dataset/Electricity/LD2011_2014.csv')
        assert dim < dataset.shape[1]
        self.data = torch.tensor(dataset.iloc[:, -dim:].values, dtype=torch.float32)
        self._normalize()
        self._split(mode)


class ElectricityH(TimeSeries):
    def __init__(self, pred_len, seq_len, dim, mode):
        super().__init__(pred_len, seq_len)
        dataset = pd.read_csv('dataset/Electricity/LD2011_2014_h.csv')
        assert dim < dataset.shape[1]
        self.data = torch.tensor(dataset.iloc[:, -dim:].values, dtype=torch.float32)
        self._normalize()
        self._split(mode)


# Electricity Transformer Temperature
class ETT(TimeSeries):
    def __init__(self, pred_len, seq_len, dim, mode):
        super().__init__(pred_len, seq_len)
        dataset = pd.read_csv('dataset/Electricity/ETTm.csv')
        assert dim < dataset.shape[1]
        self.data = torch.tensor(dataset.iloc[:, -dim:].values, dtype=torch.float32)
        self._normalize()
        self._split(mode)


class ETTH(TimeSeries):
    def __init__(self, pred_len, seq_len, dim, mode):
        super().__init__(pred_len, seq_len)
        dataset = pd.read_csv('dataset/Electricity/ETTh.csv')
        assert dim < dataset.shape[1]
        self.data = torch.tensor(dataset.iloc[:, -dim:].values, dtype=torch.float32)
        self._normalize()
        self._split(mode)


# Exchange Rate
class Exchange(TimeSeries):
    def __init__(self, pred_len, seq_len, dim, mode):
        super().__init__(pred_len, seq_len)
        dataset = pd.read_csv('dataset/Exchange/exchange_rate.csv', header=None)
        assert dim <= dataset.shape[1]
        self.data = torch.tensor(dataset.iloc[:, -dim:].values, dtype=torch.float32)
        self._normalize()
        self._split(mode)


# Queries of Web Service
class QPS(TimeSeries):
    def __init__(self, pred_len, seq_len, dim, mode):
        super().__init__(pred_len, seq_len)
        dataset = pd.read_csv('dataset/QPS/MQPS.csv')
        assert dim < dataset.shape[1]
        self.data = torch.tensor(dataset.iloc[:, -dim:].values, dtype=torch.float32)
        self._normalize()
        self._split(mode)


# Solar Power
class Solar(TimeSeries):
    def __init__(self, pred_len, seq_len, dim, mode):
        super().__init__(pred_len, seq_len)
        dataset = pd.read_csv('dataset/Solar/solar_Alabama.csv')
        assert dim < dataset.shape[1]
        self.data = torch.tensor(dataset.iloc[:, -dim:].values, dtype=torch.float32)
        self._normalize()
        self._split(mode)


class SolarH(TimeSeries):
    def __init__(self, pred_len, seq_len, dim, mode):
        super().__init__(pred_len, seq_len)
        dataset = pd.read_csv('dataset/Solar/solar_Alabama_h.csv')
        assert dim < dataset.shape[1]
        self.data = torch.tensor(dataset.iloc[:, -dim:].values, dtype=torch.float32)
        self._normalize()
        self._split(mode)


class Traffic(TimeSeries):
    def __init__(self, pred_len, seq_len, dim, mode):
        super().__init__(pred_len, seq_len)
        dataset = pd.read_csv('dataset/Traffic/PeMS.csv', delimiter=',')
        assert dim < dataset.shape[1]
        self.data = torch.tensor(dataset.iloc[:, -dim:].values, dtype=torch.float32)
        self._normalize()
        self._split(mode)


# Meteorological Observations
class Weather(TimeSeries):
    def __init__(self, pred_len, seq_len, dim, mode):
        super().__init__(pred_len, seq_len)
        dataset = pd.read_csv('dataset/Weather/mpi_roof.csv')
        assert dim < dataset.shape[1]
        self.data = torch.tensor(dataset.iloc[:, -dim:].values, dtype=torch.float32)
        self._normalize()
        self._split(mode)


class WeatherH(TimeSeries):
    def __init__(self, pred_len, seq_len, dim, mode):
        super().__init__(pred_len, seq_len)
        dataset = pd.read_csv('dataset/Weather/mpi_roof_h.csv')
        assert dim < dataset.shape[1]
        self.data = torch.tensor(dataset.iloc[:, -dim:].values, dtype=torch.float32)
        self._normalize()
        self._split(mode)


data_dict = {
    'Electricity': Electricity, 'Electricity_h': ElectricityH,
    'ETT': ETT, 'ETTh': ETTH,
    'Exchange': Exchange,
    'QPS': QPS,
    'Solar': Solar, 'Solar_h': SolarH,
    'Traffic': Traffic,
    'Weather': Weather, 'Weather_h': WeatherH,
}
