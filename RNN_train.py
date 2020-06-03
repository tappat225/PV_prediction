import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
import method
from model import RNN

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

feature = ['Current', 'Wind_speed', 'Power', 'Humidity', 'Temp', 'GHI', 'DHI', 'Wind_dir']
# 设定输入特征
input_feature = ['Wind_speed', 'Humidity', 'Temp', 'GHI']
input_feature_num = 4
# 设定目标特征
target_feature = ['Power']

# 删除功率为空的数据组
data = data.dropna(subset=['Power'])

# NAN值赋0
data = data.fillna(0)
data[data < 0] = 0

# 设定样本数目
data = data[:8640]

# 归一化
scaler = MinMaxScaler()
data[feature] = scaler.fit_transform(data[feature].to_numpy())

# 数据集分配
train_x, train_y = method.create_dataset(data, target_feature, input_feature)

rnn = RNN(input_size=input_feature_num)
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)
loss_func = nn.MSELoss()
epochs = 100
print(rnn)
print('Start training...')

for e in range(epochs):
    # 前向传播
    y_pred = rnn(train_x)
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

MODEL_PATH = 'model_rnn.pth'

torch.save(rnn, MODEL_PATH)

print('Model saved')
