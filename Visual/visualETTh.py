import torch
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']
from Loader import ETT_Hour as Data
device = torch.device('cuda', 0)

def draw_ETTh(id=2100, l_pred=24):
    name = 'Model/ETTh_' + str(l_pred) + '.pth'
    model = torch.load(name).to(device)
    data = Data(device, l_pred, l_pred*4, 1, flag='valid')
    x, y = data[id]
    res, avg = model(x.unsqueeze(0))
    y_bar = res + avg
    x = x.squeeze(-1).detach().cpu().numpy()
    y = y.squeeze(-1).detach().cpu().numpy()
    y_bar = y_bar.squeeze(0).squeeze(-1).detach().cpu().numpy()
    plt.figure(figsize=(8, 3))
    plt.plot(range(l_pred * 4), x)
    plt.plot(range(l_pred * 4, l_pred * 5), y, label='real')
    plt.plot(range(l_pred * 4, l_pred * 5), y_bar, label = 'predict')
    plt.yticks([])
    plt.tick_params(labelsize=20)
    plt.xlim([0,l_pred*5])
    plt.xticks([0,25,50,75,100])
    plt.subplots_adjust(left=0.05, right=0.95, top=0.76, bottom=0.12)
    plt.savefig('Figure/visualETTh.pdf')
    plt.show()