import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
from torch.autograd import Variable

train_data = pd.read_excel('./train_data_5W.xlsx', header=0, index_col=(0, 1))
# test_data = pd.read_excel('./test_data.xlsx', header=0, index_col=(0, 1))
features = ['Temp_Out', 'Out_Hum', 'Dew_Pt', 'Wind_Speed',
            'Wind_Dir', 'Hi_Dir', 'Wind_Chill', 'Heat_Index',
            'THW_Index', 'THSW_Index', 'Solar_Rad']
train_data = train_data[features]
input_feature = ['Temp_Out', 'Out_Hum', 'Dew_Pt', 'Wind_Speed',
                 'Wind_Dir', 'Hi_Dir', 'Wind_Chill', 'Heat_Index',
                 'THW_Index', 'THSW_Index']
target_feature = ['Solar_Rad']


#
# # 归一化
# def std_fun(data):
#     # scaler = preprocessing.StandardScaler().fit(data)
#     #
#     # data_normal = scaler.transform(data)
#     data_values = data.values
#     data_values = data_values.astype('float32')
#     data_max = np.max(data_values)
#     data_min = np.min(data_values)
#     scalar = data_max - data_min
#     # data = (data - data_min) / scalar
#     # dataset = list(map(lambda x: x / scalar, data))
#     dataset = data.apply(lambda x: (x-data_min)/scalar)
#     return dataset
#
#
# # 试验
# train_data = train_data[~train_data['Solar_Rad'].isin([0])].dropna(axis=0)
# # train_data = std_fun(train_data)
# # scalar = MinMaxScaler(feature_range=(0, 1))
# # train_data = scalar.fit_transform(train_data)
# train_data = std_fun(train_data)
#
# data_X = train_data[input_feature]
# data_Y = train_data[target_feature]
#
# x_train = torch.from_numpy(data_X.to_numpy()).float()
# y_train = torch.squeeze(torch.from_numpy(data_Y.to_numpy()).float())
#
#
# #
# # def creat_dataset(data):
# #     dataX = data.drop(target_feature, axis=1)
# #     dataY = data[target_feature]
# #     return np.array(dataX), np.array(dataY)
#
# #
# # train_data = pre_std(train_data)
# # # print(train_data)
# # data_X, data_Y = creat_dataset(train_data)
# #
#
# # y_train = y_train.reshape(y_train.shape[0], 1, y_train.shape[1])
# #
# # data_X = torch.from_numpy(data_X)
# # data_Y = torch.from_numpy(data_Y)
#
#
# # print(data_X.shape)
# x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
#
#
# class LSTMNet(nn.Module):
#
#     def __init__(self):
#         super(LSTMNet, self).__init__()
#         self.rnn = nn.LSTM(
#             input_size=10,
#             hidden_size=50,
#             num_layers=1,
#             batch_first=True,
#         )
#         self.out = nn.Sequential(
#             nn.Linear(50, 1)
#         )
#
#     def forward(self, x):
#         r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
#         out = self.out(r_out[:, -1, :])
#         # print(out.shape)
#         return out
#
#
# lstm = LSTMNet()
# optimizer = torch.optim.Adam(lstm.parameters(), lr=0.02)
# loss_func = nn.MSELoss()
# epochs = 1000
# print(lstm)
# print('Start training...')
#
# for e in range(epochs):
#     # var_x = Variable(data_X).type(torch.FloatTensor)
#     # var_y = Variable(data_Y).type(torch.FloatTensor)
#     # 前向传播
#     y_pred = lstm(x_train)
#     y_pred = torch.squeeze(y_pred)
#     loss = loss_func(y_pred, y_train)
#     # 反向传播
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if e % 200 == 0:
#         print('Epoch:{}, Loss:{:.5f}'.format(e + 1, loss.item()))
#
# plt.plot(y_pred.detach().numpy(), 'r', label='y_pred')
# plt.plot(y_train.detach().numpy(), 'b', label='y_train')
# plt.legend()
# plt.show()

# print('Model saving...')
#
# MODEL_PATH = 'model_lstm.pth'
#
# torch.save(lstm, MODEL_PATH)
#
# print('Model saved')
