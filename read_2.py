import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_excel('./train_data_exl_1000.xlsx', header=0, index_col=(0, 1))
# test_data = pd.read_excel('./test_data.xlsx', header=0, index_col=(0, 1))
features = ['Temp_Out', 'Out_Hum', 'Dew_Pt.', 'Wind_Speed',
            'Wind_Dir', 'Hi_Dir', 'Wind_Chill', 'Heat_Index',
            'THW_Index', 'THSW_Index', 'Solar_Rad.']
train_data = train_data[features]
target_feature = ['Dew_Pt.']


def pre_std(data):
    data = data.dropna()
    dataset = data.values
    dataset = dataset.astype('float32')
    max_value = np.max(dataset)
    min_value = np.min(dataset)
    scalar = max_value - min_value
    dataset = list(map(lambda x: x / scalar, dataset))
    return dataset


def creat_dataset(data):
    dataX = data.drop(target_feature, axis=1)
    dataY = data[target_feature]
    return np.array(dataX), np.array(dataY)


pre_std(train_data)
# print(train_data.shape)
data_X, data_Y = creat_dataset(train_data)

data_X = data_X.reshape(-1, len(data_X))
data_Y = data_Y.reshape(-1, len(data_Y))

data_X = torch.from_numpy(data_X)
data_Y = torch.from_numpy(data_Y)



