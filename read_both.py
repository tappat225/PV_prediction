import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from trans_form import series_to_supervised


def train_test():
    # 读取文件，包含头信息
    training_data = pd.read_excel('train_data.xlsx')
    test_data = pd.read_excel('test_data.xlsx')
    # 删掉不用的字段
    feature = ['Temp_Out', 'Out_Hum', 'Dew_Pt', 'Wind_Speed', 'Wind_Dir', 'Hi_Dir', 'Wind_Chill', 'Heat_Index',
               'THW_Index', 'THSW_Index', 'Solar_Rad']
    training_data = training_data[feature]
    test_data = test_data[feature]
    ok_y = test_data['Solar_Rad']
    # df 转 array
    values_1 = test_data.values
    values = training_data.values
    # 原始数据标准化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values[:, :9])
    scaled_1 = scaler.fit_transform(values_1[:, :9])

    # 根据过去一小时预测当前
    n_hours = 1
    n_features = scaled.shape[1]
    # 构造特征，过去三小时与当前数据集合
    reframed = series_to_supervised(scaled, n_hours, 1)
    reframed_1 = series_to_supervised(scaled_1, n_hours, 1)
    values = reframed.values
    values_1 = reframed_1.values
    # 划分训练集与测试集
    size = 0.8
    num = int(values.shape[0] * size)
    train = values
    test = values_1
    train_X, train_y = train[:num, :n_features * n_hours], train[:num, -1]
    validation_X, validation_y = train[num:, :n_features * n_hours], train[num:, -1]
    test_X, test_y = test[:, :n_features * n_hours], test[:, -1]
    # reshape 为 3D [samples, timesteps, features]，将 n_hours 看成 n 个独立的时间序列而不是一个整体的
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    validation_X = validation_X.reshape((validation_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    return train_X, train_y, test_X, test_y, validation_X, validation_y, scaler, n_hours, n_features, ok_y
