import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *

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

data = data.fillna(0)
data_rain = data
# data = data[data['Power'] > 0]
# data_rain = data[data['Rainfall'] > 0]
# data_rain = data_rain[32069:32331]
data_rain = data_rain[460:722]
data_rain = data_rain['Power']
data_rain = data_rain.values
x = np.linspace(0, 22, 262)
plt.xlabel('时间(单位：小时)')
plt.ylabel('功率(单位：Kw)')
plt.plot(x, data_rain)
plt.xlim(-1, 23)
plt.show()
