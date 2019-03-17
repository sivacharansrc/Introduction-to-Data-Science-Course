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
##
supermarket1 = df[df['Outlet_Type'] == 'Supermarket Type1']
pd.crosstab(supermarket1['Outlet_Size'], supermarket1['Outlet_Location_Type'])
##
supermarket1 = df[df.Outlet_Size.isnull()]
supermarket1.Outlet_Location_Type.value_counts()
# FE5 The above analysis suggests that all Supermarket1 Tier 3 are high, and Supermarket1 Tier 2 are Small. Also, in our data set all the missing values are from Tier 2

# ITEM FAT CONTENT
df.Item_Fat_Content.value_counts()

# FE6 The duplicate values of Item Fat Content should be fixed

df.groupby('Item_Fat_Content').Item_Outlet_Sales.mean()
df.groupby('Item_Fat_Content').Item_Outlet_Sales.sum() / df.Item_Outlet_Sales.sum()  # Though the average cost of Low and Regular Fat items are almost comparable, there is a greater chance that more of Low Fat items might be sold
df.groupby('Item_Type').Item_Outlet_Sales.sum() / df.Item_Outlet_Sales.sum()
df.groupby('Outlet_Type').Item_Outlet_Sales.sum() / df.Item_Outlet_Sales.sum()
df.groupby('Outlet_Location_Type').Item_Outlet_Sales.sum() / df.Item_Outlet_Sales.sum()
df.groupby('Outlet_Size').Item_Outlet_Sales.sum() / df.Item_Outlet_Sales.sum()
df.groupby('Item_Type').Item_Outlet_Sales.mean()
df.groupby(['Item_Fat_Content', 'Item_Type']).Item_Outlet_Sales.mean()
plt.scatter(df.Item_Weight, df.Item_Outlet_Sales) # Looks like there is not much information with Weight and Sales


# FE1 - IMPUTING THE ZERO VALUES OF ITEM VISIBILITY
minItemVisibility = (df['Item_Visibility'][df.Item_Visibility != 0]).min()
df.loc[df['Item_Visibility'] == 0, 'Item_Visibility'] = minItemVisibility

# FE2 - CUBE ROOT TRANSFORMATION TO FIX THE RIGHT SKEWNESS
df.loc[:, 'Item_Visibility'] = np.cbrt(df['Item_Visibility'])
# df.Item_Visibility.plot.box()
# The cube root transformation has fixed the distribution as well as outliers

# FE3 - IMPUTING MISSING ITEM WEIGHT BY THE MEAN OF ITEM TYPE
df['Item_Weight'] = df.groupby('Item_Type').Item_Weight.transform(lambda x: x.fillna(x.median()))

# FE4 - FIXING MISSING VALUES FOR OUTLET SIZE - ALL GROCERY STORE ARE SMALL IN SIZE, AND ALL SUPERMARKET TYPE 2 AND 3 ARE MEDIUM IN SIZE
df['Outlet_Size'] = np.where(df['Outlet_Size'].isnull(), np.where(df['Outlet_Type'] == 'Grocery Store', "Small", np.where(df['Outlet_Type'].isin(["Supermarket Type2", "Supermarket Type3"]), "Medium", df['Outlet_Size'])), df['Outlet_Size'])

# FE5 - WITHIN SUPERMARKET1, ALL TIER 3 ARE HIGH IN SIZE AND ALL TIER 2 ARE SMALL
df['Outlet_Size'] = np.where(df['Outlet_Size'].isnull(), np.where((df['Outlet_Type'] == 'Supermarket Type1') & (df['Outlet_Location_Type'] == 'Tier 3'), 'High', np.where((df['Outlet_Type'] == 'Supermarket Type1') & (df['Outlet_Location_Type'] == 'Tier 2'), 'Small', df['Outlet_Size'])), df['Outlet_Size'])


# FE6 - FIXING THE DUPLICATE VALUES OF ITEM FAT CONTENT

df.loc[df['Item_Fat_Content'].isin(['LF', 'low fat']), 'Item_Fat_Content'] = 'Low Fat'
df.loc[df['Item_Fat_Content'] == 'reg', 'Item_Fat_Content'] = 'Regular'

# FE7 - CONVERTING ALL CATEGORICAL VARIABLES TO PROPORTION
df['Fat_Content_Prop'] = df.groupby('Item_Fat_Content').Item_Outlet_Sales.transform(lambda x: x.sum()) / df.Item_Outlet_Sales.sum()
df['Item_Type_Prop'] = df.groupby('Item_Type').Item_Outlet_Sales.transform(lambda x: x.sum()) / df.Item_Outlet_Sales.sum()
df['Outlet_Location_Type_Prop'] = df.groupby('Outlet_Location_Type').Item_Outlet_Sales.transform(lambda x: x.sum()) / df.Item_Outlet_Sales.sum()
df['Outlet_Type_Prop'] = df.groupby('Outlet_Type').Item_Outlet_Sales.transform(lambda x: x.sum()) / df.Item_Outlet_Sales.sum()
df['Outlet_Size_Prop'] = df.groupby('Outlet_Size').Item_Outlet_Sales.transform(lambda x: x.sum()) / df.Item_Outlet_Sales.sum()
df['Total_Prop'] = df.groupby(['Item_Fat_Content', 'Item_Type', 'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Size']).Item_Outlet_Sales.transform(lambda x: x.sum()) / df.Item_Outlet_Sales.sum()

# FE8 - CREATING A VARIABLE THAT HAS INFORMATION ON HOW MANY YEARS THE OUTLET HAS BEEN IN BUSINESS
from datetime import datetime as dt
curr_dt = dt.now()
df['Years_In_Business'] = curr_dt.year - df['Outlet_Establishment_Year']

# FEATURE SELECTION FOR MODEL
df_final = df[['ID', 'Item_Outlet_Sales', 'Total_Prop', 'Item_MRP', 'Item_Weight',  'Fat_Content_Prop', 'Item_Type_Prop',  'Outlet_Location_Type_Prop', 'Outlet_Type_Prop',  'Outlet_Size_Prop', 'Years_In_Business', 'Item_Fat_Content', 'Item_Type', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type']]

# ENCODING CATEGORICAL VARIABLES

one_hot_encoding = ['Item_Fat_Content', 'Item_Type', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type']
df_one_hot = df_final.loc[:, one_hot_encoding]
df_final.drop(one_hot_encoding, axis=1, inplace=True)

# PERFORMING ONE HOT ENCODING
df_one_hot = pd.get_dummies(df_one_hot, drop_first=True)

# CONCATENATING ALL COLUMNS
df_final = pd.concat([df_final, df_one_hot], axis=1)


# SEGREGATING THE TRAIN AND TEST
train = df_final[0:len(df_train)]
test = df_final[len(df_train):len(df_final)]

y_train = train['Item_Outlet_Sales']
x_train = train.drop(['ID', 'Item_Outlet_Sales'], axis=1)

test_ID = test['ID']
test = test.drop(['ID', 'Item_Outlet_Sales'], axis=1)

# SPLITTING TRAIN DATA SET IN TO TRAIN AND VALIDATION
from sklearn.model_selection import train_test_split

x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size= 0.2)

# FITTING A LINEAR REGRESSION MODEL
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

ols = LinearRegression()
ols.fit(x_train, y_train)
ols.score(x_train, y_train)

predictions = ols.predict(x_validation)
r2_score(y_validation, predictions)

# The model is only able to explain 40% of the variance

# PREDICTING THE TEST DATA SET

predictions = ols.predict(test)
output = pd.DataFrame({'ID': test_ID, 'Item_Outlet_Sales': predictions})
output.to_csv("C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course/Regression Project/output/Submission 2 - Linear Regression with 55 percent r square.csv", index=False, header=True)