import pandas as pd

messages_df = pd.read_csv('../../temp/test1.csv', sep=',', index_col=0)
print(messages_df.info())
print(messages_df.describe())
print(messages_df.head())