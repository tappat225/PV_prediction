import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import method
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from model import BiLSTMNet
from sklearn.metrics import mean_squared_error


file_path = './91-Site_1A-Trina_5W.csv'
data = pd.read_csv(file_path, header=0, low_memory=False, index_col=0)
data = data.rename(columns={
    u'1A Trina - Active Energy Delivered-Received (kWh)': 'AE_Power',
    u'1A Trina - Current Phase Average (A)': 'Current', #电流
    u'1A Trina - Wind Speed (m/s)': 'Wind_speed',   #风速
    u'1A Trina - Active Power (kW)': 'Power',   #功率
    u'1A Trina - Weather Relative Humidity (%)': 'Humidity',    #湿度
    u'1A Trina - Weather Temperature Celsius (\xb0C)': 'Temp',    #气温
    u'1A Trina - Global Horizontal Radiation (W/m\xb2)': 'GHI',   #全球水平辐照度
    u'1A Trina - Diffuse Horizontal Radiation (W/m\xb2)': 'DHI',   #扩散水平辐照度
    u'1A Trina - Wind Direction (Degrees)': 'Wind_dir',  #风向
    u'1A Trina - Weather Daily Rainfall (mm)': 'Rainfall'   #降雨
})
data = data.drop(columns='AE_Power')
data = data.drop(columns='Rainfall')
input_feature_num = 7
feature = ['Current', 'Wind_speed', 'Power', 'Humidity', 'Temp', 'GHI', 'DHI', 'Wind_dir']
# 设定输入特征
input_feature = ['Wind_speed', 'Humidity', 'Temp', 'GHI']
# 设定目标特征
target_feature = ['Power']
# dataset = data[~data['Power'].isin([0])].dropna(axis=0)

# 删除功率为空的数据组
# data = data.dropna(subset=['Power'])
# NAN值赋0
data = data.fillna(0)
data[data < 0] = 0

# 归一化
scaler = MinMaxScaler()
data[feature] = scaler.fit_transform(data[feature].to_numpy())

# 设置因素特征数据集
data_x = data[input_feature]
# 设置目标特征数据集
data_y = data[target_feature]
# data.to_csv('./dataset.csv')

test_x = data_x[10000:10864]
test_y = data_y[10000:10864]


# 格式转为numpy
test_x = torch.from_numpy(test_x.to_numpy()).float()
test_y = torch.squeeze(torch.from_numpy(test_y.to_numpy()).float())
# x转tensor
test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1])

print('data ready')
print('testing...')
bi_lstm = torch.load('model_bi_lstm.pth')
# print(lstm)
bi_lstm = bi_lstm.eval()
prediction = bi_lstm(test_x)
prediction = prediction.view(-1).data.numpy()
MSE = mean_squared_error(test_y, prediction)
# R_2 = 1-MSE/test_y


print('drawing...')
plt.plot(prediction, 'r', label='prediction')
plt.plot(test_y, 'b', label='real')
plt.legend(loc='best')
plt.show()
print('Complete!')

print('MSE: ', MSE)
# print('R^2: ', R_2)
