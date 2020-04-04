import torch
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE_PATH = './test_data.xlsx'
df = pd.read_excel(CSV_FILE_PATH, header=0, index_col=(0, 1))
data_set = df
# print(df.head(5))


groups = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
i = 1
plt.figure(figsize=(10, 10))
for group in groups:
    values = data_set.values
    data_set = data_set[abs((values[:, group] - values[:, group].mean()) / values[:, group].std()) < 3]
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group])
    plt.title(data_set.columns[group], y=0.5, loc='right')
    i += 1
plt.show()