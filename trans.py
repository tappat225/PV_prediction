import pandas as pd

pdata = pd.read_excel('./test_data.xlsx')
# df = pdata.to_csv('test_data.csv')
print(pdata.head(5))