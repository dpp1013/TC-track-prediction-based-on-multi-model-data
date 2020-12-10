import warnings
import os, sys

warnings.filterwarnings('ignore')
from model_RNN import RNN
from model_LSTM import lstm
from model_seq2seq import Seq2SeqRnn

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
import evaluate_function_com
from evaluate_function_com import DataSet
start = time.time()
data = pd.read_csv('../dataset/24_predicted_24_supervised_data.csv')
lr = 1e-3
epoches = 100
criterion = nn.MSELoss()
weight_decays = 1e-3
batch_sizes = 64

data_new = data.iloc[:, 1:]

data_train, data_test = train_test_split(data_new, test_size=0.2, shuffle=True)
train_data = DataSet(data_train,8)

test_data = DataSet(data_test,8)

train_dataloader = DataLoader(dataset=train_data, batch_size=batch_sizes, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_sizes, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_weights(m):
    for name, parm in m.named_parameters():
        #         print(name,parm)
        nn.init.uniform_(parm.data, -0.08, 0.08)


n_features = 6
seq_length = 8
label_length = 8
# model = Seq2SeqRnn(input_size = 6, seq_len = 8, hidden_size = 100, output_size = 2 )
model = lstm()
model.apply(init_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decays)

if __name__ == '__main__':
    test_loss = []
    train_loss = []
    for epoch in range(epoches):
        model = model.train()
        total_loss = 0
        t = tqdm(train_dataloader, leave=False, total=len(train_dataloader))
        for batch_idx, data in enumerate(t):
            x, y = data
            # print('初始x shape，y shape',x.shape,y.shape)
            x = x.to(device).view(-1, seq_length, n_features)
            y = y.to(device).view(-1, label_length, 2)
            # print('之后x shape，y shape', x.shape, y.shape)
            optimizer.zero_grad()
            output = model(x)
            # print(output.shape)
            loss = criterion(output, y)
            loss_aver = loss.item()
            t.set_postfix({
                'trainloss': '{:.8f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })
            loss.backward()
            optimizer.step()
            total_loss += loss.cpu().item()
        test_loss_one, eval_list = evaluate_function_com.evaulate(model, test_dataloader, criterion)
        print('epoch:{},训练集 loss:{}, 测试集 loss:{}, 测试 MAE：{}, MSE:{}, RMSE:{},R2_lat:{},R2_lon:{},APE:{}'.format(
            epoch + 1,
            total_loss / len(train_dataloader),
            test_loss_one,
            eval_list[0],
            eval_list[1],
            eval_list[2],
            eval_list[3],
            eval_list[4],
            eval_list[5]
            ))
        test_loss.append(test_loss_one)
        train_loss.append(total_loss / len(train_dataloader))
    # 可视化
    evaluate_function_com.loss_show(test_loss, train_loss, epoches)
    evaluate_function_com.save_model(model)
print(test_loss)
print(train_loss)
end = time.time()
print('总时间time：',(end-start))