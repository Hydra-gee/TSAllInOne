import torch
from torch.utils.data import Dataset
import pandas as pd


class TimeSeries(Dataset):
    def __init__(self, pred_len: int, seq_len: int) -> None:
        self.pred_len = pred_len
        self.seq_len = seq_len

    def _normalize(self) -> None:
        avg = torch.mean(self.data, dim=0, keepdim=True)
        std = torch.std(self.data, dim=0, keepdim=True)
        self.data = (self.data - avg) / std

    def _split(self, mode: str) -> None:
        total_len = self.data.shape[0] - self.seq_len - self.pred_len * 3 + 3
        max_train_idx = int(total_len * 0.7) + self.seq_len + self.pred_len - 1
        max_valid_idx = int(total_len * 0.8) + self.seq_len + self.pred_len * 2 - 2
        if mode == 'train':
            self.data = self.data[:max_train_idx]
        elif mode == 'valid':
            self.data = self.data[max_train_idx + self.pred_len - 1: max_valid_idx]
        elif mode == 'test':
            self.data = self.data[max_train_idx + self.pred_len - 1:]

    def __len__(self) -> int:
        return self.data.shape[0] - self.seq_len - self.pred_len + 1

    def __getitem__(self, item: int) -> (torch.Tensor, torch.Tensor):
        left, mid, right = item, item + self.seq_len, item + self.seq_len + self.pred_len
        x = self.data[left: mid]
        y = self.data[mid: right]
        return x, y


class TSDataset(TimeSeries):
    def __init__(self, pred_len: int, seq_len: int, dim: int, path: str, device: torch.device, mode: str) -> None:
        super().__init__(pred_len, seq_len)
        dataset = pd.read_csv(path)
        assert dim < dataset.shape[1]
        self.data = torch.Tensor(dataset.iloc[:, -dim:].values).to(device)
        self._normalize()
        self._split(mode)


# paths of csv files, [hourly sampled, original dataset]
data_path_dict = {
    'Electricity': ['dataset/Electricity/LD2011_2014_h.csv', 'dataset/Electricity/LD2011_2014.csv'],  # Electricity Consumption
    'ETT': ['dataset/ETT/ETTh.csv', 'dataset/ETT/ETTm.csv'],  # Electricity Transformer Temperature
    'Exchange': ['dataset/Exchange/exchange_rate.csv', 'dataset/Exchange/exchange_rate.csv'],  # Exchange Rate
    'QPS': ['dataset/QPS/HQPS.csv', 'dataset/QPS/MQPS.csv'],  # Queries of Web Service
    'Solar': ['dataset/Solar/solar_Alabama_h.csv',  'dataset/Solar/solar_Alabama.csv'],  # Solar Power
    'Traffic': ['dataset/Traffic/PeMS.csv', 'dataset/Traffic/PeMS.csv'],
    'Weather': ['dataset/Weather/mpi_roof_h.csv', 'dataset/Weather/mpi_roof.csv']  # Meteorological Observations
}
