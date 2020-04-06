import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

train_data = pd.read_excel('./train_data_exl_1000.xlsx', header=0, index_col=(0, 1))
# test_data = pd.read_excel('./test_data.xlsx', header=0, index_col=(0, 1))
features = ['Temp_Out', 'Out_Hum', 'Dew_Pt.', 'Wind_Speed',
            'Wind_Dir', 'Hi_Dir', 'Wind_Chill', 'Heat_Index',
            'THW_Index', 'THSW_Index', 'Solar_Rad.']
train_data = train_data[features]
target_feature = ['Dew_Pt.']


# 标准化
def pre_std(data):
    data = data.dropna()
    dataset = data.values
    dataset = dataset.astype('float32')
    max_value = np.max(dataset)
    min_value = np.min(dataset)
    scalar = max_value - min_value
    dataset = list(map(lambda x: x / scalar, dataset))
    return dataset


def creat_dataset(data):
    dataX = data.drop(target_feature, axis=1)
    dataY = data[target_feature]
    return np.array(dataX), np.array(dataY)


pre_std(train_data)
# print(train_data.shape)
data_X, data_Y = creat_dataset(train_data)

data_X = data_X.reshape(data_X.shape[0], 1, data_X.shape[1])
data_Y = data_Y.reshape(data_Y.shape[0], 1, data_Y.shape[1])

data_X = torch.from_numpy(data_X)
data_Y = torch.from_numpy(data_Y)
print(data_X.shape)


class LSTMNet(nn.Module):

    def __init__(self):
        super(LSTMNet, self).__init__()
        self.rnn = nn.LSTM(
            input_size=10,
            hidden_size=50,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.Linear(50, 1)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out[:, -1, :])
        # print(out.shape)
        return out


lstm = LSTMNet()
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.02)
loss_func = nn.MSELoss()
epochs = 500
print(lstm)

for e in range(epochs):
    var_x = Variable(data_X).type(torch.FloatTensor)
    var_y = Variable(data_Y).type(torch.FloatTensor)
    # 前向传播
    out = lstm(var_x)
    loss = loss_func(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (e+1) % 25 == 0:
        print('Epoch:{}, Loss:{:.5f}'.format(e+1, loss.item()))

