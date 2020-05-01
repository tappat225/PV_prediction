import numpy as np


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



