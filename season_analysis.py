import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']
file_path = './91-Site_1A-Trina_10W.csv'
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

data = data['Power']

# NAN值赋0
data = data.fillna(0)
data[data < 0] = 0


# 设定样本数目
# data = data[23500:24364] #Random
data_winter = data[2407:2695] # Winter
data_spring = data[13927:14215] # Spring
data_summer = data[41260:41548] # Summer
data_autumn = data[70056:70344] # Autumn

data_spring = data_spring.values
data_summer = data_summer.values
data_autumn = data_autumn.values
data_winter = data_winter.values


x = np.linspace(0, 24, 288)
plt.xlabel('时间(单位：小时)')
plt.ylabel('功率(单位：Kw)')
plt.plot(x, data_spring, 'green', label='春季')
# plt.title('春季')
plt.plot(x, data_summer, 'gold', label='夏季')
# plt.title('夏季')
plt.plot(x, data_autumn, 'brown', label='秋季')
# plt.title('秋季')
plt.plot(x, data_winter, 'aqua', label='冬季')
# plt.title('冬季')
plt.xlim(0, 25)
plt.legend(loc='upper right')
plt.show()
