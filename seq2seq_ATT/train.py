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
start = time.time()
data = pd.read_csv('../dataset/72_predicted_72_supervised_data.csv')

lr = 1e-3
epoches = 100
criterion = nn.MSELoss()
weight_decays = 1e-3
batch_sizes = 64
data_new = data.iloc[:, 1:]
print(data_new.shape)
data_train, data_test = train_test_split(data_new, test_size=0.2, shuffle=True)
train_data = DataSet_seq(data_train,24)
test_data = DataSet_seq(data_test,24)

train_dataloader = DataLoader(dataset=train_data, batch_size=batch_sizes, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_sizes, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_weights(m):
    for name, parm in m.named_parameters():
        #         print(name,parm)
        nn.init.uniform_(parm.data, -0.08, 0.08)


n_features = 6
seq_length = 24
label_length = 24

model = Seq2Seq(seq_len=24, n_features=6, embedding_dim=128, output_length=24)
# init_weights(model)
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
            x = x.to(device).view(-1, seq_length, n_features)
            y = y.to(device).view(-1, label_length, 2)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.permute(1, 0, 2), y)
            loss_aver = loss.item()
            t.set_postfix({
                'trainloss': '{:.8f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })
            loss.backward()
            optimizer.step()
            total_loss += loss.cpu().item()
        test_loss_one, eval_list = evaluate_function.evaulate(model, test_dataloader, criterion)
        print('epoch:{},训练集 loss:{}, 测试集 loss:{}, 反归一化后 结果 测试 MAE：{}, MSE:{}, RMSE:{},R2_lat:{},R2_lon:{},APE:{}'.format(
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
    evaluate_function.loss_show(test_loss, train_loss, epoches)
    evaluate_function.save_model(model)
print(test_loss)
print(train_loss)
end = time.time()
print('总时间time：',(end-start))