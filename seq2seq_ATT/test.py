'''
画R2
'''

import warnings
import os, sys

warnings.filterwarnings('ignore')
from Seq2Seq_att import DataSet_seq, Seq2Seq
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import numpy as np
import math
import time

os.chdir(os.path.dirname(__file__))
sys.path.append("..")
import evaluate_function
import matplotlib.pyplot as plt

start = time.time()
data = pd.read_csv('../dataset/96_predicted_48_supervised_data.csv')

lr = 1e-3
epoches = 100
criterion = nn.MSELoss()
weight_decays = 1e-3
batch_sizes = 32
data_new = data.iloc[:, 1:]
data_train, data_test = train_test_split(data_new, test_size=0.2, shuffle=True)
test_data = DataSet_seq(data_test, 32)

test_dataloader = DataLoader(dataset=test_data, batch_size=batch_sizes, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load('48_48_Seq2Seq_att_model.pkl')
seq_length = 32
n_features = 6
label_length = 32


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


def evaulate(model, test_dataloader):
    with torch.no_grad():
        a = 0
        for i, data in enumerate(test_dataloader):
            x, y = data
            x = x.to(device).view(-1, seq_length, n_features)
            y = y.to(device).view(-1, label_length, 2)
            # print(x.shape)
            # print(y.shape)
            output_test = model(x)
            # print('测试集输出 output_test',output_test)
            # rnn测试不需要,但是seq2seq需要
            output_test = output_test.permute(1, 0, 2)
            unnormal(output_test)
            unnormal(y)
            a += 1
            if a ==6:
                return output_test, y


pred, real = evaulate(model, test_dataloader)
pred_lat = np.array(pred[:, :, 0:1]).reshape(-1)
pred_lon = np.array(pred[:, :, 1:]).reshape(-1)
real_lat = np.array(real[:, :, 0:1]).reshape(-1)
real_lon = np.array(real[:, :, 1:]).reshape(-1)

print(pred_lat)
print(real_lat)
# x = np.arange(100,170)
x = np.arange(0,64)
y = x
# plt.ylabel('Prediction latitude value')
# plt.xlabel('Observsed latitude value')
plt.ylabel('prediction longitude value')
plt.xlabel('Observsed longitude value')

plt.plot(x,y)
plt.scatter(pred_lon, real_lon, s=10, marker='*',label = 'R2= 0.85 ')

# plt.scatter(pred_lat, real_lat, s=10, marker='*',label = 'R2= 0.81 ')
plt.legend()
plt.show()