#encoding:utf-8
import pandas as pd
import pandas_profiling

df = pd.read_excel('train_data_exl_1000.xlsx')
# pandas_profiling.ProfileReport(df)
print(df.shape)


