import torch.nn as nn
import torch

from torch import nn
from torch.autograd import Variable


class lstm(nn.Module):
    def __init__(self, input_size=6, hidden_size=100, output_size=2, num_layer=3):
        super(lstm, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.layer1(x)
        s, b, h = x.size()
        x = x.view(s * b, h)
        x = self.layer2(x)
        x = x.view(s, b, -1)
        return x

model = lstm()
print(model)
ts = torch.randn((32,8,6))
out = model(ts)
print(out.shape)