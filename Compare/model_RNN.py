'''
RNN模型
'''
import torch
import torch.nn as nn
import numpy as np

TIME_STEP = 16  # rnn time step
INPUT_SIZE = 4  # rnn input size
LR = 1e-3  # learning rate


# LSTM
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(6, 6, 3)  # 输入数据2个特征维度，6个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果
        self.out = nn.Linear(6, 2)  # 线性拟合，接收数据的维度为6，输出数据的维度为1

    def forward(self, x):
        x1, _ = self.lstm(x)
        a, b, c = x1.shape
        out = self.out(x1.view(-1, c))
        out1 = out.view(a, b, -1)
        return out1


LSTM = RNN()
print(LSTM)
test1 = torch.randn((32, 8, 6))
out1 = LSTM(test1)
print(out1.shape)
