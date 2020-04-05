import torch
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE_PATH = './test.csv'
df = pd.read_csv(CSV_FILE_PATH, header=0)
date_set = df.head(1000)
# print(df.head(5))

values = date_set.values
groups = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
i = 1
plt.figure(figsize=(10, 10))
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group])
    plt.title(date_set.columns[group], y=0.5, loc='right')
    i += 1
plt.show()

# values = dataset.values  
# # specify columns to plot  
# groups = [0, 1, 2, 3, 5, 6, 7, 8, 9]  
# i = 1
# # plot each column  
#     pyplot.figure(figsize=(10,10))  
# for group in groups:  
#    pyplot.subplot(len(groups), 1, i)  
#     pyplot.plot(values[:, group])  
#   pyplot.title(dataset.columns[group], y=0.5, loc='right')  
# i+= 1
# plt.show()


# v2.0 #
# values = train_data.values
# groups = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# i = 1
# plt.figure(figsize=(10, 10))
# for group in groups:
#     plt.subplot(len(groups), 1, i)
#     plt.plot(values[:, group])
#     plt.title(train_data.columns[group], y=0.5, loc='right')
#     i += 1
# plt.show()
