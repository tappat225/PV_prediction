import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import concat
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

############
# 导入数据
############
# CSV_FILE_PATH = './train_data.xlsx'
# df = pd.read_excel(CSV_FILE_PATH, header=0, index_col=(0, 1))
# data = df
# data.head()

#############
# 数据预处理
##############
# features = ['Temp_Out', 'Out_Hum', 'Dew_Pt', 'Wind_Speed',
#             'Wind_Dir', 'Hi_Dir', 'Wind_Chill', 'Heat_Index',
#             'THW_Index', 'THSW_Index', 'Solar_Rad']
# train_data = data.head(100)
# 去除坏值
# groups = [0, 1, 2, 3, 5, 6, 7, 8, 9]
# for group in groups:
#     values = train_data.values
#     train_data = train_data[abs((values[:, group] - values[:, group].mean()) / values[:, group].std()) < 3]
# # 标准化
# scaler = MinMaxScaler(feature_range=(-1, 1))
# scaled = scaler.fit_transform(values[:, :9])
# train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))
# print(train_data_normalized[:5])
#

# # 设定参数
# n_hours = 1
# n_features = scaled.shape[1]
#
# reframed = series_to_supervised(scaled, n_hours, 1)
# values = reframed.values
# train = values
#
#
#############
# 搭建LSTM模型
#############
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
