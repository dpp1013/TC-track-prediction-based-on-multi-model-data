# Seq2Seq_Atten

import torch
import torch.nn.functional as F
import time
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device is:", device)


class DataSet_seq(Dataset):

    def __init__(self, data_pd,step):
        data = np.array(data_pd)
        self.x_data = torch.from_numpy(data[:, 0:step * 6]).type(torch.float32)
        self.y_data = torch.from_numpy(data[:, step * 6:]).type(torch.float32)
        self.len = data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=128):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, embedding_dim
        self.num_layers = 3
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=3,
            batch_first=True,
            # dropout=0.35
        )

    def forward(self, x):
        # x = x.reshape((1, self.seq_len, self.n_features))

        h_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim).to(device))

        c_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim).to(device))

        x, (hidden, cell) = self.rnn1(x, (h_1, c_1))

        # return hidden_n.reshape((self.n_features, self.embedding_dim))
        return x, hidden, cell


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()
        # super().__init__()

        self.attn = nn.Linear((enc_hid_dim) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        # 拿出最后一个number_layer: ht
        hidden = hidden[2:3, :, :].permute(1, 0, 2)

        src_len = encoder_outputs.shape[1]

        # print("hidden size is",hidden.size())

        # repeat decoder hidden state src_len times
        # hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        hidden = hidden.repeat(1, src_len, 1)

        # encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # print("encode_outputs size after permute is:",encoder_outputs.size())

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]
        a = torch.cat((hidden, encoder_outputs), dim=2)
        _attn = self.attn(a)
        energy = torch.tanh(_attn)

        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=128, n_features=1):
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = input_dim, n_features

        self.rnn1 = nn.LSTM(
            input_size=1,
            hidden_size=input_dim,
            num_layers=3,
            batch_first=True,
            # dropout=0.35
        )

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x, input_hidden, input_cell):
        x = x.reshape((1, 1, 1))

        x, (hidden_n, cell_n) = self.rnn1(x, (input_hidden, input_cell))

        x = self.output_layer(x)
        return x, hidden_n, cell_n


class AttentionDecoder(nn.Module):
    def __init__(self, seq_len, attention, input_dim=128, n_features=2, encoder_hidden_state=128):
        super(AttentionDecoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = input_dim, n_features
        self.attention = attention

        self.rnn1 = nn.LSTM(
            # input_size=1,
            input_size=encoder_hidden_state + 2,  # Encoder Hidden State + One Previous input
            hidden_size=input_dim,
            num_layers=3,
            batch_first=True,
            # dropout=0.35
        )
        self.output_layer = nn.Linear(self.hidden_dim * 2, 2)

    def forward(self, x, input_hidden, input_cell, encoder_outputs):
        a = self.attention(input_hidden, encoder_outputs)
        a = a.unsqueeze(1)
        # print(a.shape)
        # a = [batch size, 1, src len]
        # encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch size, src len, enc hid dim * 2]
        weighted = torch.bmm(a, encoder_outputs)
        # print('x.shape', x.shape)
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, 2)
        rnn_input = torch.cat((x, weighted), dim=2)
        # print('rnn_input.shape', rnn_input.shape)
        # x, (hidden_n, cell_n) = self.rnn1(x,(input_hidden,input_cell))
        x, (hidden_n, cell_n) = self.rnn1(rnn_input, (input_hidden, input_cell))
        output = x.squeeze(0)
        weighted = weighted.squeeze(0)
        _out = torch.cat((output, weighted), dim=1)
        x = self.output_layer(_out.view(-1, self.hidden_dim * 2))

        return x, hidden_n, cell_n


class Seq2Seq(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=128, output_length=24):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.attention = Attention(128, 128)
        self.output_length = output_length
        self.decoder = AttentionDecoder(seq_len, self.attention, embedding_dim, n_features).to(device)

    def forward(self, x, prev_y=None):
        # print('x.shape', x.shape)
        encoder_output, hidden, cell = self.encoder(x)
        # Prepare place holder for decoder output
        targets_ta = []
        # prev_output become the next input to the LSTM cell
        x_batch = x.shape[0]

        if prev_y is None:
            prev_y = torch.zeros((x_batch, 2)).to(device)
        # print('prev_y')
        # print(prev_y)
        prev_output = prev_y

        # itearate over LSTM - according to the required output days
        for out_days in range(self.output_length):
            prev_x, prev_hidden, prev_cell = self.decoder(prev_output, hidden, cell, encoder_output)
            hidden, cell = prev_hidden, prev_cell
            prev_output = prev_x
            # print('prev_output shape', prev_output.shape)
            # print(prev_output)
            # print('prev_x.resdhape(1)', prev_x.reshape(2))

            targets_ta.append(prev_x.reshape(-1,2))
            # targets_ta.append(prev_x)
        targets = torch.stack(targets_ta)
        # print('target shape',targets.shape)
        # print(targets)

        return targets


# 输入步长16 输入特征数量6 内部特征维度512 预测步长数8
model = Seq2Seq(seq_len=16, n_features=6, embedding_dim=3, output_length=8)

print(model)
