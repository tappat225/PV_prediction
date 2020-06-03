import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
import method
from model import *
from pylab import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import  mean_absolute_error

mpl.rcParams['font.sans-serif'] = ['SimHei']
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
# data = data.dropna(subset=['Power'])

# NAN值赋0
data = data.fillna(0)
data[data < 0] = 0

# 设定样本数目
data = data[23500:24364] #Random
# data = data[2119:2983] # Winter
# data = data[13927:14791] # Spring
# data = data[40684:41548] # Summer
# data = data[69480:70344] # Autumn
# 找零值点
# zero_poi = data.isin[0]

# 归一化
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
data[input_feature] = x_scaler.fit_transform(data[input_feature].to_numpy())
data[target_feature] = y_scaler.fit_transform(data[target_feature].to_numpy())

# 数据集分配
test_x, test_y = method.create_dataset(data, target_feature, input_feature)

# 导入模型
BP_Net = torch.load('model_bp.pth')
RNN = torch.load('model_rnn.pth')
LSTM = torch.load('model_lstm.pth')
BiLSTM = torch.load('model_bi_lstm.pth')


def prediction(model, series_x, series_y, name):
    model = model.eval()
    pred = model(series_x)
    pred[pred < 0] = 0
    length = len(series_y)
    for i in range(length):
        if series_y[i] == 0:
            pred[i] = 0
    pred = pred.view(-1).data.numpy()
    pred = y_scaler.inverse_transform(pred.reshape(-1, 1))
    series_y = y_scaler.inverse_transform(series_y.reshape(-1, 1))
    MSE = mean_squared_error(series_y, pred)
    RMSE = sqrt(MSE)
    R2 = r2_score(series_y, pred)
    MAE = mean_absolute_error(series_y, pred)
    # MAPE = method.MAPE_value(series_y, pred)
    print(name, ' :')
    print(' MSE: {:.3f}'.format(MSE))
    print(' RMSE: {:.3f}'.format(RMSE))
    print(' MAE: {:.3f}'.format(MAE))
    print(' R2: {:.3f}'.format(R2))
    # print(' MAPE: {:.5f}%'.format(MAPE))
    return pred


pred_bp = prediction(BP_Net, test_x, test_y, 'BP_Net')
pred_rnn = prediction(RNN, test_x, test_y, 'RNN')
pred_lstm = prediction(LSTM, test_x, test_y, 'LSTM')
pred_bilstm = prediction(BiLSTM, test_x, test_y, 'BiLSTM')
test_y = y_scaler.inverse_transform(test_y.reshape(-1, 1))

print('Drawing...')
# ax = plt.gca()
# #去掉边框
# ax.spines['top'].set_color('none')
# ax.spines['right'].set_color('none')
# #移位置 设为原点相交
# ax.xaxis.set_ticks_position('bottom')
# ax.spines['bottom'].set_position(('data', 0))
# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data', 0))
x = np.linspace(0, 72, 864)
# plt.plot(x, pred_bp, 'aqua', label='BP预测值')
# plt.plot(x, pred_rnn, 'brown', label='RNN预测值')
# plt.plot(x, pred_lstm, 'gold', label='LSTM预测值')
plt.plot(x, pred_bilstm, 'green', label='BILSTM预测值')
plt.plot(x, test_y, 'r', label='实际值')
# plt.title('模式效果对比')
# plt.title('日期:2013(8/22,8/23,8/24)') # Winter
# plt.title('日期:2013(10/2,10/3,10/4)') # Spring
# plt.title('日期:2014(1/3,1/4,1/5)') # Summer
# plt.title('日期:2014(4/13,4/14,4/15)') # Autumn
# plt.title('BPNN72小时预测')
# plt.title('RNN72小时预测')
# plt.title('LSTM72小时预测')
plt.title('Bi-LSTM 72小时预测')
plt.xlabel('时间(单位：小时)')
plt.ylabel('功率(单位：kW)')
plt.xlim(0, 73)
plt.legend(loc='upper right')
plt.show()
print('Done')
