import pandas as pd
import numpy as np
import torch.nn as nn
from  sklearn.model_selection import train_test_split
from model_seq2seq import DataSet_seq,Seq2SeqRnn
from torch.utils.data import DataLoader,dataset
import torch
import time
import os
import sys
os.chdir(os.path.dirname(__file__))
sys.path.append("..")
import evaluate_function


data = pd.read_csv('../dataset/supervised_data.csv')
data = data.replace({np.nan: 0.0})

# 超参数设置
lr = 1e-4
epoches = 100
criterion = nn.MSELoss(size_average=True)
weight_decays = 1e-3
batch_sizes = 128
start = time.time()

# 整体数据
data_new = data.iloc[16:, 1:]
# 准备数据
data_train, data_test = train_test_split(data_new, test_size=0.2, shuffle=True)
train_data = DataSet_seq(data_train)
test_data = DataSet_seq(data_test)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_sizes, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_sizes, shuffle=False)
# 权重初始化
def init_weights(m):
    for name, parm in m.named_parameters():
        nn.init.uniform_(parm.data, -0.08, 0.08)

# 参数设置
n_features = 6  # 时刻的特征 6
seq_length = 16  # 输入步长
label_length = 8  # 输出步长
model = Seq2SeqRnn(input_size=6, seq_len=16, hidden_size=64, output_size=16, num_layers=1, bidirectional=False)
print('模型结构', model)
model.apply(init_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decays)
# 训练

for epoch in range(epoches):
    model = model.train()
    train_loss = []
    test_loss = []
    for batch_idx, data in enumerate(train_dataloader):
        x, y = data
        x = x.view(-1, 16, 6)
        y = y.view(-1, 1, 16)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    train_loss.append(loss.item())
    # 测试
    model.eval()
    test_loss_one,eval_list = evaluate_function.evaulate(model, test_dataloader, loss)
    print('epoch:{},训练集 loss:{},训练集 loss:{}'.format(epoch + 1, loss.item(),test_loss))
    print('测试 MAE：{}, MSE:{}, RMSE:{},LAT_R2:{},LON_R2:{}'.format(
        eval_list[0],eval_list[1],eval_list[2],eval_list[3],eval_list[4]))
    test_loss.append(test_loss_one)
# 可视化
evaluate_function.loss_show(test_loss, train_loss, epoches)
end = time.time()
times = end - start
print(times)