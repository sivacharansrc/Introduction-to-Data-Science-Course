import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('expand_frame_repr', False)
import numpy as np
import matplotlib.pyplot as plt

# read train and test set
train = pd.read_csv("C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course/Data Files/Predictive Modeling and Machine Learning/train_titanic.csv")
test = pd.read_csv("C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course/Data Files/Predictive Modeling and Machine Learning/test_titanic.csv")

# combining train and test dataset
df = (train.append(test, ignore_index=True)).reset_index(drop=True)

# check missing values
df.isnull().sum()

# remove Cabin variable
df.drop('Cabin', axis=1, inplace=True)

# fill missing values in Age and Embarked variables
df['Age'].fillna(df['Age'].median(), inplace = True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)

# check missing values again
df.isnull().sum()

# remove Ticket and Name variables
df.drop(['Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder

# label enconding
label = LabelEncoder()

# label encode Embarked and Sex variables
df['Embarked_num'] = label.fit_transform(df['Embarked'])
df['Sex_num'] = label.fit_transform(df['Sex'])

# drop Embarked and Sex variables
df = df.drop(['Sex', 'Embarked'], axis = 1)

# see description of the dataframe df
df.describe()

df['Parch'].value_counts()

df['Pclass'].value_counts()

df['Sex_num'].value_counts()

df['Age'].plot.hist()

df['Age'].plot.box()

df['Fare'].plot.hist()

# scatter plot between Age and Fare
df.plot.scatter('Age', 'Fare')

# bar plot between Pclass and mean Age
df.groupby('Pclass')['Age'].mean().plot.bar()

# bar plot between Pclass and mean Fare
df.groupby('Pclass')['Fare'].mean().plot.bar()

train = df[:len(train)]
test = df[len(train):]

true_val = test['Survived']

# delete Survived variable from test
test.drop(['Survived'], axis=1, inplace=True)

# remove rows where Fare is 400 or above
train = train[train['Fare']<400]

# Replace the outliers in Fare with its mean. The outliers are approximately above the value 62.
train.loc[train['Fare']> 62, 'Fare'] = np.mean(train['Fare'])

# Replace the outliers in Age variable with its median. The outliers are approximately above the value 55.
train.loc[train['Age']> 55, 'Age'] = np.median(train['Age'])

# remove Survived from train
xtrain = train.drop('Survived', axis = 1)

ytrain = train['Survived']

from sklearn.linear_model import LogisticRegression

lreg = LogisticRegression(n_jobs=1, multi_class='ovr', solver='liblinear')

# fit the logistic regression model lreg
lreg.fit(xtrain, ytrain)

# make prediction on the test dataset
pred = lreg.predict(test)

# evaluate the model lreg
lreg.score(test, true_val)

# 0.81142857142857139