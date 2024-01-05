import torch
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']
from data_loader import ETT_Minute as Data
device = torch.device('cuda', 0)

def draw_ETTm(pos=200, l_pred=96):
    name = 'files/ETTm_'+str(l_pred) + '.pth'
    model = torch.load(name).to(device)
    data = Data(device, l_pred, l_pred*4, 1, flag='valid')
    x, y = data[pos]
    res, avg = model(x.unsqueeze(0))
    y_bar = res + avg
    x = x.squeeze(-1).detach().cpu().numpy()
    y = y.squeeze(-1).detach().cpu().numpy()
    y_bar = y_bar.squeeze(0).squeeze(-1).detach().cpu().numpy()
    plt.figure(figsize=(8, 3))
    plt.plot(range(l_pred * 4), x, label='History')
    plt.plot(range(l_pred * 4, l_pred * 5), y, label='Real')
    plt.plot(range(l_pred * 4, l_pred * 5), y_bar, label = 'Prediction')
    plt.legend(bbox_to_anchor=(0.5, 1.2), loc=10, fontsize=24, ncol=3)
    plt.yticks([])
    plt.tick_params(labelsize=20)
    plt.xlim([0,l_pred*5])
    #plt.xticks([0,25,50,75,100])
    plt.subplots_adjust(left=0.05, right=0.95, top=0.76, bottom=0.12)
    plt.savefig('Figure/visualETTm.pdf')
    plt.show()