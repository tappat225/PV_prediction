import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing


# 归一化[0,1]
# def std_fun(data):
#     data_values = data.values
#     data_values = data_values.astype('float32')
#     data_max = np.max(data_values)
#     data_min = np.min(data_values)
#     scalar = data_max - data_min
#     dataset = data.apply(lambda x: (x-data_min)/scalar)
#     return dataset


# 位规范化
# def point_move(data):
#     data_values = data.values
#     data_values = data_values.astype('float32')
#     data_max = np.max(data_values)
#     poi = -abs(math.log10(data_max))
#     dataset = data.apply(lambda x: x*(10**poi))
#     return dataset


# 使用库归一化[0,1]
def skl_std_fun(data, features):
    scaler = preprocessing.MinMaxScaler()
    # 标准化训练集数据
    data[features] = scaler.fit_transform(data[features].to_numpy())
    return data


# 标准化[-1,1]
def nor_fun(data):
    data_values = data.values
    data_values = data_values.astype('float32')
    data_max = np.max(data_values)
    data_min = np.min(data_values)
    # scalar = data_max - data_min
    scalar = np.std(data_values)
    average = np.mean(data_values)
    dataset = data.apply(lambda x: (x - average) / scalar)
    return dataset


# 删除目标特征的零值
def data_wash(data, target_feature):
    # dataset = data[~data['Solar_Rad'].isin([0])].dropna(axis=0)
    dataset = data[~data[target_feature].isin([0])].dropna(axis=0)
    return dataset


# 数据集调整，X转tensor，y转一维序列
def create_dataset(data, target_features, input_features):
    data_x = data[input_features]
    data_y = data[target_features]
    data_x = torch.from_numpy(data_x.to_numpy()).float()
    data_x = data_x.reshape(data_x.shape[0], 1, data_x.shape[1])
    data_y = torch.squeeze(torch.from_numpy(data_y.to_numpy()).float())
    return data_x, data_y


# 箱形图异常值处理
def box_plot_del(data):
    Percentile = np.percentile(data, [0, 25, 50, 75, 100])
    IQR = Percentile[3] - Percentile[1]
    UpLimit = Percentile[3] + IQR * 1.5
    DownLimit = Percentile[1] - IQR * 1.5


# 箱形图查看
def box_plot_view(data, target_feature):
    Percentile = np.percentile(data[target_feature], [0, 25, 50, 75, 100])
    IQR = Percentile[3] - Percentile[1]
    UpLimit = Percentile[3] + IQR * 1.5
    DownLimit = Percentile[1] - IQR * 1.5


# 画正态分布图
def plot_normal_distribution(data, target_feature):
    data_values = data[target_feature].values
    mean = np.mean(data_values)
    sigma = np.std(data_values)
    y = np.exp(-((data_values - mean) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    x = np.linspace(mean-3*sigma, mean+3*sigma, 26546)
    plt.plot(x, y)
    # 画出直方图，最后的“normed”参数，是赋范的意思，数学概念
    plt.hist(x, bins=10, rwidth=0.9, normed=True)
    plt.title('Time distribution')
    plt.xlabel('value')
    plt.ylabel('Probability')
    # 输出
    plt.show()


# MAE误差
# def MAE_value(y_true, y_pred):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     n = len(y_true)
#     mae = sum(np.abs(y_true - y_pred)) / n
#     return mae


# MAPE误差
def MAPE_value(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
