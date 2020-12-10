# 参数 + 评价指标
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from torch.utils.data import Dataset
class DataSet(Dataset):

    def __init__(self, data_pd,step):
        data = np.array(data_pd)
        self.x_data = torch.from_numpy(data[:, 0:step * 6]).type(torch.float32)
        self.y_data = torch.from_numpy(data[:, step * 6:]).type(torch.float32)
        self.len = data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


n_features = 6
seq_length = 24
label_length = 24
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def to_radians(x, *y):
    return math.radians(x)

def distance(pred, real):
    '''
    :param pred:  shape (n,2)
    :param real:  shape (n,2)
    :return: (n,1)
    '''
    R = 6357
    latPred = pred[:,:, 0:1].squeeze(-1)
    latPred.map_(latPred,to_radians)
    lonPred = pred[:,: ,1:].squeeze(-1)
    lonPred.map_(lonPred,to_radians)
    latReal = real[:,:, 0:1].squeeze(-1)
    latReal.map_(latPred,to_radians)
    lonReal = real[:,:, 1:].squeeze(-1)
    lonReal.map_(lonReal,to_radians)

    E1 = 2 * R * torch.asin(
        torch.sqrt(
            torch.sin(
                torch.pow((latPred - latReal) / 2, 2)
            )
            + torch.cos(latReal) * torch.cos(latPred) *
            torch.sin(torch.pow((lonPred - lonReal) / 2, 2))
        )
    )
    E2 = 2 * R * torch.asin(
        torch.sqrt(
            torch.pow(torch.sin((latPred - latReal) / 2), 2) + torch.cos(latReal) * torch.cos(latPred) * torch.pow(
                (lonPred - lonReal) / 2, 2)
        )
    )

    return torch.mean(E1)


def unnormal(data):
    '''
    :param data: 归一化后的数据
    :return: 反归一化
    LAT_MAX = 179.96099999999998 LAT_MIN = 100.0
    LON_MAX = 62.1 LON_MAX = 0.6
    '''
    LAT_MAX = 179.96099999999998
    LAT_MIN = 100.0
    LON_MAX = 62.1
    LON_MIN = 0.6
    data[:, :, 0:1] = data[:, :, 0:1] * (LAT_MAX - LAT_MIN) + LAT_MIN
    data[:, :, 1:] = data[:, :, 1:] * (LON_MAX - LON_MIN) + LON_MIN
    return data


def mae_mse_rmse(target, prediction):
    '''
    :param target: (batch,label_len,features)
    :param prediction:
    :return:
    '''
    # print('target shape', target.shape)
    n = len(target) * 2
    # 反归一化后的数据
    target = unnormal(target)
    prediction = unnormal(prediction)
    # print('反归一化后的target',target[0:1,])
    # print('反归一化后的目标prediction',prediction[0:1,])
    E1 = distance(target,prediction)

    lat_real = np.array(target[:, :, 0:1].squeeze(-1))
    lon_real = np.array(target[:, :, 1:].squeeze(-1))
    lat_pred = np.array(prediction[:, :, 0:1].squeeze(-1))
    lon_pred = np.array(prediction[:, :, 1:].squeeze(-1))
    lat_mae = mean_absolute_error(lat_pred, lat_real)
    lon_mae = mean_absolute_error(lon_pred, lon_real)
    mae = (lat_mae + lon_mae) / 2
    lat_mse = mean_squared_error(lat_pred, lat_real)
    lon_mse = mean_squared_error(lon_pred, lon_real)
    mse = (lat_mse + lon_mse) / 2
    lat_r2 = r2_score(lat_pred, lat_real)
    lon_r2 = r2_score(lon_pred, lon_real)
    rmse = mse ** 0.5
    return mae, mse, rmse, lat_r2, lon_r2,E1

def evaulate(model, test_dataloader, loss_func):
    model.eval()
    # it = iter(test_dataloader)
    total_count = 0
    total_loss = 0
    total_mae = 0
    total_mse = 0
    total_rmse = 0
    total_R2_lat = 0
    total_R2_lon = 0
    total_E1 = 0
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            x, y = data
            x = x.to(device).view(-1, seq_length, n_features)
            y = y.to(device).view(-1, label_length, 2)
            output_test = model(x)
            # print('测试集输出 output_test',output_test)
            # rnn测试不需要,但是seq2seq需要
            output_test = output_test.permute(1, 0, 2)
            with torch.no_grad():
                loss = loss_func(output_test, y)
                mae, mse, rmse, R2_lat, R2_lon,E1 = mae_mse_rmse(output_test, y)
                total_count += 1
                total_loss += loss.item()
                total_mae += mae
                total_mse += mse
                total_rmse += rmse
                total_R2_lat += R2_lat
                total_R2_lon += R2_lon
                total_E1 += E1
    model.train()
    return total_loss / total_count, [
        total_mae / total_count,
        total_mse / total_count,
        total_rmse / total_count,
        total_R2_lat / total_count,
        total_R2_lon / total_count,
        total_E1 / total_count
    ]


def loss_show(test_loss, train_loss, epoches):
    x_epoch = np.arange(1, epoches + 1)
    plt.figure()
    plt.plot(x_epoch, train_loss, label='train_loss')
    plt.plot(x_epoch, test_loss, label='test_loss')
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def save_model(model):
    torch.save(model, "./72_72_Seq2Seq_att_model.pkl")

#
# if __name__ == '__main__':
#     pred = torch.randn((32, 8, 2))
#     real = torch.randn((32, 8, 2))
#     mae, mse, rmse, R2_lat, R2_lon = mae_mse_rmse(pred, real)
#     print(mae, mse, rmse, R2_lat, R2_lon)
