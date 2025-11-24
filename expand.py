import numpy as np
import pandas as pd


# Import and sort
df = pd.read_csv("PIE_data.csv", sep=";", header=0, index_col=0)
df = df.sort_values(by=['F_SESSION', 'F_START_FRAME'], ascending=[True, True])

df['F_DURATION'] = pd.to_timedelta(df['F_DURATION'])
df['F_DURATION_sec'] = df['F_DURATION'].dt.total_seconds()

# Repeat rows based on 'quantity'
md = 1
df['quantity'] = np.ceil(df['F_DURATION_sec']/md)

# Repeats the index quantity times, then select rows based on that repeated index
df_expanded = df.loc[df.index.repeat(df['quantity'])]

# Cleanup
df_expanded = df_expanded.reset_index(drop=True)
df_expanded = df_expanded.drop('quantity', axis=1)

df_expanded.to_csv('transformed_data.csv')