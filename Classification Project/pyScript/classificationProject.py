import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('expand_frame_repr', False)
import matplotlib.pyplot as plt
import datetime as dt

# https://datahack.analyticsvidhya.com/contest/introduction-to-data-science-classification/

### Reading the data set

df = pd.read_csv("C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course/Classification Project/input/Train.csv")
df.head()

df.isnull().sum()
#1 From the above, it is evident that there are certain rows where all values related to manager is missing. All such rows should be removed from the data set
df.shape

#2 The ID column is not going to be helpful in predictions. So, the column can be totally removed from the data set
#3 The Application Receipt date can be further broken down to month and year, and can then be removed
#4 Convert Applicant Birthdate and Manager Birthdate to Age


df_clean.isnull().sum()



#1 Remove all the rows where all variables related to manager is missing
df_clean = df[~df['Manager_DOJ'].isnull()]

#2 Removing the ID Column
df_clean.drop('ID', axis=1, inplace=True)

#3 Feature Engineering Application Recipt Date and Manager DOJ
df_clean['Application_Receipt_Date'] = pd.to_datetime(df_clean['Application_Receipt_Date'])
df_clean['Application_Receipt_Year'] = df_clean['Application_Receipt_Date'].dt.strftime("%Y")
df_clean['Application_Receipt_Month'] = df_clean['Application_Receipt_Date'].dt.strftime("%m")
df_clean.drop('Application_Receipt_Date', axis=1, inplace=True)

#4 Converting Applicant and Manager Birthdate to Age instead of date (i.e. Age as of Jan 1 2009)
df_clean['Applicant_BirthDate'] = pd.to_datetime(df_clean['Applicant_BirthDate'])
x =  dt.datetime(2009, 1, 1,0,0)
df_clean['Applicant_Age'] = (x - df_clean['Applicant_BirthDate'])
df_clean['Applicant_Age'] = df_clean['Applicant_Age'].dt.days / 365

# Manager DOB
df_clean['Manager_DoB'] = pd.to_datetime(df_clean['Manager_DoB'])
x =  dt.datetime(2009, 1, 1,0,0)
df_clean['Manager_Age'] = (x - df_clean['Manager_DoB'])
df_clean['Manager_Age'] = df_clean['Manager_Age'].dt.days / 365

df_clean.drop(['Manager_DoB', 'Applicant_BirthDate'], axis=1, inplace=True)


df_clean.head()