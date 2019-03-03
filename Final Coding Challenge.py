import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('expand_frame_repr', False)

proj_dir = "C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course/"
path = proj_dir + "Data Files/Predictive Modeling and Machine Learning/titanic.csv"
df = pd.read_csv(path)

# Number of columns and rows
df.shape

# How many columns as int64
df.dtypes[df.dtypes == 'int64']
df.dtypes.value_counts()

# Missing values in Age
df['Age'].isnull().value_counts()

# Mean of Age
df['Age'].mean()

# Mean Age of Male Passengers
df['Age'][df['Sex'] == 'male'].mean()

# Correlation between Fare and Survived

df.corr()

# Percentage of female that survived

len(df[(df['Sex'] == 'female') & (df['Survived'] == 1)]) / len(df[(df['Sex'] == 'female')])

# Number of passengers who survived and who did not having PClass as 2

df['Survived'][df['Pclass'] == 2].value_counts()

# Median fare of the ticket for PClass 1, 2, 3
df.groupby('Pclass').agg({'Fare': 'median'})

# Missing Values in Embarked

df['Embarked'].head()

# Percentage of missing values in Cabin
df['Cabin'].isnull().value_counts()[1] / df['Cabin'].__len__()

# 95th percentile of Fare variable
df['Fare'].quantile(0.95)

# Number of passengers paid more than 250
df[df['Fare'] > 250].__len__()

# Distribution of Fare is right Skewed?
df['Fare'].plot.hist()

# Does Age and Fare have Outliers? Use box plot - Ans: Both have outliers

df[['Fare', 'Age']].plot.box()