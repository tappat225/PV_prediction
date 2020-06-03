import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import method
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model import Net_BP


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


bp_net = Net_BP(n_features=input_feature_num)

optimizer = torch.optim.SGD(bp_net.parameters(), lr=0.01)
loss_func = nn.MSELoss()
epochs = 2000
print(bp_net)
print('Start training...')

for e in range(epochs):
    # 前向传播
    y_pred = bp_net(train_x)
    y_pred = torch.squeeze(y_pred)
    loss = loss_func(y_pred, train_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if e % 400 == 0:
        print('Epoch:{}, Loss:{:.5f}'.format(e, loss.item()))

plt.plot(y_pred.detach().numpy(), 'r', label='y_pred')
plt.plot(train_y.detach().numpy(), 'b', label='y_train')
plt.legend()
plt.show()

print('Done.')

print('Model saving...')

MODEL_PATH = 'model_bp.pth'

torch.save(bp_net, MODEL_PATH)

print('Model saved')
#
# # 测试
# test_x = data_x[25000:]
# test_y = data_y[25000:]
#
# # 格式转为numpy
# test_x = torch.from_numpy(test_x.to_numpy()).float()
# test_y = torch.squeeze(torch.from_numpy(test_y.to_numpy()).float())
# # x转tensor
# test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1])
#
# bp_net.eval()
# prediction = bp_net(test_x)
# prediction = prediction.view(-1).data.numpy()
# MSE = mean_squared_error(test_y, prediction)
# # R_2 = 1-MSE/test_y
#
#
# print('drawing...')
# plt.plot(prediction, 'r', label='prediction')
# plt.plot(test_y, 'b', label='real')
# plt.legend(loc='best')
# plt.show()
# print('Complete!')
#
# print('MSE: ', MSE)
# # print('R^2: ', R_2)