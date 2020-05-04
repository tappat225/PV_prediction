import numpy as np
import torch


# 归一化
def std_fun(data):
    data_values = data.values
    data_values = data_values.astype('float32')
    data_max = np.max(data_values)
    data_min = np.min(data_values)
    scalar = data_max - data_min
    dataset = data.apply(lambda x: (x-data_min)/scalar)
    return dataset


# 数据清洗
def data_wash(data):
    dataset = data[~data['Solar_Rad'].isin([0])].dropna(axis=0)
    return dataset


# 数据集调整
def create_dataset(data, target_features, input_features):
    data_x = data[input_features]
    data_y = data[target_features]
    data_x = torch.from_numpy(data_x.to_numpy()).float()
    data_x = data_x.reshape(data_x.shape[0], 1, data_x.shape[1])
    data_y = torch.squeeze(torch.from_numpy(data_y.to_numpy()).float())
    return data_x, data_y

