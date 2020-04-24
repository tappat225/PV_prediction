import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


train_data = pd.read_excel('./train_data_5W.xlsx', header=0, index_col=(0, 1))
# test_data = pd.read_excel('./test_data.xlsx', header=0, index_col=(0, 1))
features = ['Temp_Out', 'Out_Hum', 'Dew_Pt', 'Wind_Speed',
            'Wind_Dir', 'Hi_Dir', 'Wind_Chill', 'Heat_Index',
            'THW_Index', 'THSW_Index', 'Solar_Rad']
train_data = train_data[features]
target_feature = ['Solar_Rad']
train_data = train_data[~train_data['Solar_Rad'].isin([0])].dropna(axis=0)
print(train_data.shape)
#
# # 标准化
# def pre_std(data):
#
#     dataset = data.values
#     dataset = dataset.astype('float32')
#     max_value = np.max(dataset)
#     min_value = np.min(dataset)
#     scalar = max_value - min_value
#     dataset = list(map(lambda x: x / scalar, dataset))
#     return dataset
#
# #
# # def creat_dataset(data):
# #     dataX = data.drop(target_feature, axis=1)
# #     dataY = data[target_feature]
# #     return np.array(dataX), np.array(dataY)
#
#
# train_data = pre_std(train_data)

