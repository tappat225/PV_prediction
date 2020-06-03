import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import method
from model import LSTMNet
from sklearn.metrics import mean_squared_error

# test_data = pd.read_excel('./train_data_5W.xlsx', header=0, index_col=(0, 1))
test_data = pd.read_excel('./test_data.xlsx', header=0, index_col=(0, 1))
features = ['Temp_Out', 'Out_Hum', 'Dew_Pt', 'Wind_Speed',
            'Wind_Dir', 'Hi_Dir', 'Wind_Chill', 'Heat_Index',
            'THW_Index', 'THSW_Index', 'Solar_Rad']
test_data = test_data[features]
input_feature = ['Temp_Out', 'Out_Hum', 'Dew_Pt', 'Wind_Speed',
                 'Wind_Dir', 'Hi_Dir', 'Wind_Chill', 'Heat_Index',
                 'THW_Index', 'THSW_Index']
target_feature = ['Solar_Rad']

print('data transforming...')

# 数据清洗
test_data = method.data_wash(test_data)
# 提取3000组作为测试
test_data = test_data[:3000]
# print(test_data.shape)
test_data = method.std_fun(test_data)

# 确立测试数据集
test_x, test_y = method.create_dataset(test_data, target_feature, input_feature)

print('data ready')
print('testing...')

lstm = torch.load('model_lstm.pth')
# print(lstm)
lstm = lstm.eval()
prediction = lstm(test_x)
prediction = prediction.view(-1).data.numpy()
MSE = mean_squared_error(test_y, prediction)
R_2 = 1-MSE/test_y


print('drawing...')
plt.plot(prediction, 'r', label='prediction')
plt.plot(test_y, 'b', label='real')
plt.legend(loc='best')
plt.show()
print('Complete!')

print('MSE: ', MSE)
print('R^2: ', R_2)
