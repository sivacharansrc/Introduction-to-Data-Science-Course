import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('expand_frame_repr', False)
import numpy as np


# Reading Training Data Set:
df_train = pd.read_csv("C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course/Regression Project/input/train.csv")
df_test = pd.read_csv("C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course/Regression Project/input/test.csv")
df_test['Item_Outlet_Sales'] = np.nan
df = pd.concat([df_train, df_test], axis=0, sort=True) # 5045 Test data Set rows

# EXPLORATORY ANALYSIS
import matplotlib.pyplot as plt
from scipy import math

# ITEM VISIBILITY
plt.hist(df['Item_Visibility'], bins=20)

# The visibility of an item cannot be 0. Hence, replacing zero with minimum
minItemVisibility = (df['Item_Visibility'][df.Item_Visibility != 0]).min()
df.loc[df['Item_Visibility'] == 0, 'Item_Visibility'] = minItemVisibility

plt.subplot(2,2,1)
df.Item_Visibility.plot.hist()
plt.title = "Item_Visibility"
plt.subplot(2,2,2)
np.log10(df.Item_Visibility).plot.hist()
plt.title = 'Log10'
plt.subplot(2,2,3)
np.sqrt(df.Item_Visibility).plot.hist()
plt.title = 'SQRT'
plt.subplot(2,2,4)
np.cbrt(df.Item_Visibility).plot.hist()
plt.title = 'CBRT'
plt.show()

# The distribution of the Item Visibility looks much better with the cube root
df.loc[:, 'Item_Visibility'] = np.cbrt(df['Item_Visibility'])
df.Item_Visibility.plot.box()

# The cube root transformation has fixed the distribution as well as outliers


# ITEM FAT CONTENT
df.Item_Fat_Content.value_counts()
# Duplicate values are factors are available. This needs to be fixed

df.loc[df['Item_Fat_Content'].isin(['LF', 'low fat']), 'Item_Fat_Content'] = 'Low Fat'
df.loc[df['Item_Fat_Content'] == 'reg', 'Item_Fat_Content'] = 'Regular'

# Fixed duplicate factors for Fat Item Content

# OUTLET SIZE - FIXING MISSING VALUES
df.drop('Outlet_Identifier', axis=1, inplace=True)