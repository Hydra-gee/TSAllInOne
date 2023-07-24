from torch.utils.data import Dataset
from torch import mean, std, Tensor
from pandas import read_csv
from numpy import loadtxt


class TimeSeries(Dataset):
    def __init__(self, l_pred, l_seq, d_in):
        self.pred_len = l_pred
        self.seq_len = l_seq
        self.input_dim = d_in
        self.data = None
        self.dev = None
        self.avg = None

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

    def normalize(self):
        self.avg = mean(self.data, dim=0, keepdim=True)
        self.dev = std(self.data, dim=0, keepdim=True)
        self.data = (self.data - self.avg) / self.dev

    def split(self, flag):
        if flag == 'train':
            self.data = self.data[:int(self.data.shape[0] * 0.6), -self.input_dim:]
        elif flag == 'valid':
            self.data = self.data[int(self.data.shape[0] * 0.6):int(self.data.shape[0] * 0.8), -self.input_dim:]
        elif flag == 'test':
            self.data = self.data[int(self.data.shape[0] * 0.8):, -self.input_dim:]


class ETT_Hour(TimeSeries):
    def __init__(self, device, l_pred, l_seq, d_in, flag='train'):
        super().__init__(l_pred, l_seq, d_in)
        dataset = read_csv('DataCSV/ETT/ETTh1.csv')
        assert d_in < dataset.shape[1]
        self.data = Tensor(dataset.iloc[:, 1:].values).to(device)
        self.normalize()
        self.split(flag)


class ETT_Minute(TimeSeries):
    def __init__(self, device, l_pred, l_seq, d_in, flag='train'):
        super().__init__(l_pred, l_seq, d_in)
        dataset = read_csv('DataCSV/ETT/ETTm1.csv')
        assert d_in < dataset.shape[1]
        self.data = Tensor(dataset.iloc[:, 1:].values).to(device)
        self.normalize()
        self.split(flag)


class Electricity(TimeSeries):
    def __init__(self, device, l_pred, l_seq, d_in, flag='train'):
        super().__init__(l_pred, l_seq, d_in)
        dataset = read_csv('DataCSV/Electricity/LD2011_2014.csv')
        assert d_in < dataset.shape[1]
        self.data = Tensor(dataset.iloc[:, 11:11+d_in].values).to(device)
        self.normalize()
        self.split(flag)



class Exchange(TimeSeries):
    def __init__(self, device, l_pred, l_seq, d_in, flag='train'):
        super().__init__(l_pred, l_seq, d_in)
        dataset = loadtxt('DataCSV/ExchangeRate/exchange_rate.csv', delimiter=',')
        assert d_in <= dataset.shape[1]
        self.data = Tensor(dataset).to(device)
        self.normalize()
        self.split(flag)


class Solar(TimeSeries):
    def __init__(self, device, l_pred, l_seq, d_in, flag='train'):
        super().__init__(l_pred, l_seq, d_in)
        dataset = loadtxt('DataCSV/SolarEnergy/solar_AL.csv', delimiter=',')
        assert d_in <= dataset.shape[1]
        self.data = Tensor(dataset[:, -d_in:]).to(device)
        self.normalize()
        self.split(flag)


class Traffic(TimeSeries):
    def __init__(self, device, l_pred, l_seq, d_in, flag='train'):
        super().__init__(l_pred, l_seq, d_in)
        dataset = loadtxt('DataCSV/Traffic/traffic.csv', delimiter=',')
        assert d_in <= dataset.shape[1]
        self.data = Tensor(dataset[:, -d_in:]).to(device)
        self.normalize()
        self.split(flag)


class Weather(TimeSeries):
    def __init__(self, device, l_pred, l_seq, d_in, flag='train'):
        super().__init__(l_pred, l_seq, d_in)
        dataset = read_csv('DataCSV/Weather/weather.csv', encoding='ISO-8859-1')
        self.data = Tensor(dataset.iloc[:, -d_in:].values).to(device)
        self.normalize()
        self.split(flag)


class Stock(TimeSeries):
    def __init__(self, device, l_pred, l_seq, d_in, flag='train'):
        super().__init__(l_pred, l_seq, d_in)
        dataset = read_csv('DataCSV/Stock/BEN.csv')
        self.data = Tensor(dataset['Close'].values).unsqueeze(-1).to(device)
        self.normalize()
        self.split(flag)

class QPS(TimeSeries):
    def __init__(self, device, l_pred, l_seq, d_in, flag='train'):
        super().__init__(l_pred, l_seq, d_in)
        dataset = read_csv('DataCSV/MQPS/app2.csv')
        self.data = Tensor(dataset['y'].values).unsqueeze(-1).to(device)
        self.normalize()
        self.split(flag)