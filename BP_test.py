import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import method
import matplotlib.pyplot as plt
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
data = data.drop(columns='AE_Power')
data = data.drop(columns='Rainfall')
feature_num = 7

# 设定输入特征
input_feature = ['Current', 'Wind_speed', 'Humidity', 'Temp', 'GHI', 'DHI', 'Wind_dir']
# 设定目标特征
target_feature = ['Power']

# 删除功率为空的数据组
data = data.dropna(subset=['Power'])

# NAN值赋0
data = data.fillna(0)

# 归一化
data = method.std_fun(data)

# 设置因素特征数据集
data_x = data.drop(columns=target_feature)

# 设置目标特征数据集
data_y = data[target_feature]

test_x = data_x[25000:]
test_y = data_y[25000:]


BP_net = torch.load('model_bp.pth')
print(BP_net)
BP_net = BP_net.eval()
prediction = BP_net(test_x)
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
