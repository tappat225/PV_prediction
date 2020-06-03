import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import method
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from model import BiLSTMNet


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
# data = data.drop(columns='AE_Power')
# data = data.drop(columns='Rainfall')

feature = ['Current', 'Wind_speed', 'Power', 'Humidity', 'Temp', 'GHI', 'DHI', 'Wind_dir']
# 设定输入特征
input_feature = ['Wind_speed', 'Humidity', 'Temp', 'GHI']
input_feature_num = 4
# 设定目标特征
target_feature = ['Power']
# dataset = data[~data['Power'].isin([0])].dropna(axis=0)

# 删除功率为空的数据组
data = data.dropna(subset=['Power'])
# NAN值赋0
data = data.fillna(0)
data[data < 0] = 0
# data = data > 0
# 归一化

scaler = MinMaxScaler()
data[feature] = scaler.fit_transform(data[feature].to_numpy())
# print(data['Power'].describe())
# 设置因素特征数据集
data_x = data[input_feature]
# 设置目标特征数据集
data_y = data[target_feature]
# data.to_csv('./dataset.csv')

train_x = data_x[:8640]
train_y = data_y[:8640]


#########
# 画图查看特征元素的曲线
########
# i = 1
# p = np.arange(1, 8, 1)
# plt.figure(figsize=(10, 10))
# for i in p:
#     plt.subplot(len(p), 1, i)
#     plt.plot(data.values[:, i])
#     plt.title(data.columns[i], y=0.5, loc='right')
#     i += 1
#
# plt.show()
#
# 格式转为numpy
train_x = torch.from_numpy(train_x.to_numpy()).float()
train_y = torch.squeeze(torch.from_numpy(train_y.to_numpy()).float())
# x转tensor
train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
# 导入网络模型
bi_lstm = BiLSTMNet(input_size=input_feature_num)
# bi_lstm = nn.LSTM(input_size=8, hidden_size=64, num_layers=2, bidirectional=True, batch_first=True)
# bi_lstm = RNN_BI(input_dim=input_feature_num)

optimizer = torch.optim.Adam(bi_lstm.parameters(), lr=0.01)
loss_func = nn.MSELoss()
epochs = 100
print(bi_lstm)
print('Start training...')

for e in range(epochs):

    # 前向传播
    y_pred = bi_lstm(train_x)
    y_pred = torch.squeeze(y_pred)
    loss = loss_func(y_pred, train_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if e % 20 == 0:
        print('Epoch:{}, Loss:{:.5f}'.format(e, loss.item()))

plt.plot(y_pred.detach().numpy(), 'r', label='y_pred')
plt.plot(train_y.detach().numpy(), 'b', label='y_train')
plt.legend()
plt.show()

print('Model saving...')

MODEL_PATH = 'model_bi_lstm.pth'

torch.save(bi_lstm, MODEL_PATH)

print('Model saved')
