import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('expand_frame_repr', False)
import numpy as np
import matplotlib.pyplot as plt

# Reading Training Data Set:
df_train = pd.read_csv("C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course/Regression Project/input/train.csv")
df_test = pd.read_csv("C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course/Regression Project/input/test.csv")
df_test['Item_Outlet_Sales'] = np.nan
df = pd.concat([df_train, df_test], axis=0, sort=True) # 5045 Test data Set rows

# EXPLORATORY ANALYSIS

# EXAMINING THE NULL VALUES
df.isnull().sum() * 100 / len(df)
### 17% of Item_Weight is missing, and # 29% of Outlet Size is missing

# EXAMINING OUTLIERS
plt.subplot(421)
df['Item_MRP'].plot.box()
plt.subplot(422)
df['Item_MRP'].plot.hist()
plt.subplot(423)
df['Item_Visibility'].plot.box()
plt.subplot(424)
df['Item_Visibility'].plot.hist()
plt.subplot(425)
df['Item_Weight'].plot.box()
plt.subplot(426)
df['Item_Weight'].plot.hist()
plt.subplot(427)
df['Outlet_Establishment_Year'].plot.box()
plt.subplot(428)
df['Outlet_Establishment_Year'].plot.hist()
plt.show()

# The Item_Visibility has lot of outliers, and the distribution is also right skewed. We can apply some form of transformation to Item_Visibility
# FE1: Also, the minimum visibility cannot be zero. So, let us impute all zeros to min of the Item Visibility (ignoring zeros)

# ITEM VISIBILITY

plt.subplot(221)
df.Item_Visibility.plot.hist()
plt.xlabel('Item_Visibility')
plt.subplot(222)
np.log10(df.Item_Visibility).plot.hist()
plt.xlabel('Log10')
plt.subplot(223)
np.sqrt(df.Item_Visibility).plot.hist()
plt.xlabel('SQRT')
plt.subplot(224)
np.cbrt(df.Item_Visibility).plot.hist()
plt.xlabel('CBRT')
plt.show()
# FE2 - Either the Square Root Transformation or the Cube Root Transformation should work fine

# ANALYZING ITEM WEIGHT TO FIX MISSING VALUES
df[(~df['Item_Weight'].isnull()) & (df['Outlet_Establishment_Year'] == 1985)].head(10)
df[(df['Item_Weight'].isnull()) & (df['Outlet_Establishment_Year'] == 1985)].head(10)
df.groupby(['Item_Type', 'Outlet_Establishment_Year']).agg({'Item_Weight': 'mean'})
# From the above analysis, all outlets with Establishment Year 1985 have their Item_Weight missing
# FE3 The mean or median Item Weight by Item Type can be used to impute missing values

# ANALYZING OUTLET SIZE TO FIX MISSING VALUES
df[(df['Outlet_Size'].isnull())].head(10)
pd.crosstab(df['Outlet_Type'], df['Outlet_Size'])
# FE4 From the above analysis, all grocery store are small, and all supermarket type 2 and 3 are Medium. Let us fix those values first







# FE1 - IMPUTING THE ZERO VALUES OF ITEM VISIBILITY
minItemVisibility = (df['Item_Visibility'][df.Item_Visibility != 0]).min()
df.loc[df['Item_Visibility'] == 0, 'Item_Visibility'] = minItemVisibility

# FE2 - CUBE ROOT TRANSFORMATION TO FIX THE RIGHT SKEWNESS
df.loc[:, 'Item_Visibility'] = np.cbrt(df['Item_Visibility'])
df.Item_Visibility.plot.box()
# The cube root transformation has fixed the distribution as well as outliers

# FE3 - IMPUTING MISSING ITEM WEIGHT BY THE MEAN OF ITEM TYPE
df['Item_Weight'] = df.groupby('Item_Type').Item_Weight.transform(lambda x: x.fillna(x.median()))

# FE4
df['Outlet_Size'] = np.where(df['Outlet_Type'] == 'Grocery Store', "Small", np.where(df['Outlet_Type'].isin(["Supermarket Type2", "Supermarket Type3"]), "Medium", df['Outlet_Size']))

# ITEM FAT CONTENT
df.Item_Fat_Content.value_counts()
# Duplicate values are factors are available. This needs to be fixed

df.loc[df['Item_Fat_Content'].isin(['LF', 'low fat']), 'Item_Fat_Content'] = 'Low Fat'
df.loc[df['Item_Fat_Content'] == 'reg', 'Item_Fat_Content'] = 'Regular'

# Fixed duplicate factors for Fat Item Content

# OUTLET SIZE - FIXING MISSING VALUES
df.drop('Outlet_Identifier', axis=1, inplace=True)