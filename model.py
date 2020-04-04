import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

CSV_FILE_PATH = './test_data.xlsx'
df = pd.read_excel(CSV_FILE_PATH, header=0, index_col=(0, 1))
data = df
data.head()

# 数据处理
features = ['Temp_Out', 'Out_Hum', 'Dew_Pt', 'Wind_Speed',
            'Wind_Dir', 'Hi_Dir', 'Wind_Chill', 'Heat_Index',
            'THW_Index', 'THSW_Index', 'Solar_Rad']
train_data = data[features]
values = train_data.values
scaler = MinMaxScaler.feature_range(0, 1)  # 标准化
