import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('expand_frame_repr', False)
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm


df = pd.read_csv("C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course/Classification Project/input/Test.csv")
df.head()
df_clean = df

#1 Remove all the rows where all variables related to manager is missing
df_clean = df[~df['Manager_DOJ'].isnull()]

#2 Removing the ID Column
#df_clean = df_clean.drop('ID', axis=1)

#3a Filling Applicant_Gender missing values with mode
df_clean.loc[df_clean['Applicant_Gender'].isnull(), 'Applicant_Gender'] = df_clean['Applicant_Gender'].mode()[0]

#3b Creating Applicant_Age from the birth date, and imputing missing data with median
df_clean['Applicant_BirthDate'] = pd.to_datetime(df_clean['Applicant_BirthDate'])
df_clean['Application_Receipt_Date'] = pd.to_datetime(df_clean['Application_Receipt_Date'])
df_clean['Applicant_Age_At_Joining'] = (df_clean['Application_Receipt_Date'] - df_clean['Applicant_BirthDate'])
df_clean['Applicant_Age_At_Joining'] = df_clean['Applicant_Age_At_Joining'].dt.days / 365
df_clean.loc[df_clean['Applicant_Age_At_Joining'].isnull(), 'Applicant_Age_At_Joining'] = df_clean['Applicant_Age_At_Joining'].median()
df_clean['Applicant_Age_At_Joining'] = df_clean['Applicant_Age_At_Joining'].astype(int)

#3c Filling Applicant_Marital_Status < 25 as S and > 25 as M

df_clean.loc[(df_clean['Applicant_Age_At_Joining'] < 25) & (df_clean['Applicant_Marital_Status'].isnull()), 'Applicant_Marital_Status'] = "S"
df_clean.loc[(df_clean['Applicant_Age_At_Joining'] >= 25) & (df_clean['Applicant_Marital_Status'].isnull()), 'Applicant_Marital_Status'] = "M"

#3d Filling Applicant_Occupation missing values as missing

df_clean.loc[df_clean['Applicant_Occupation'].isnull(), 'Applicant_Occupation'] = "Missing"

#3e Fixing the Applicant_Qualification - Recategorizing,

df_clean.loc[(~df_clean['Applicant_Qualification'].isin(['Class XII', 'Class X', 'Graduate', 'Masters of Business Administration'])) & (~df_clean['Applicant_Qualification'].isnull()), 'Applicant_Qualification'] = "Others"
df_clean.loc[df_clean['Applicant_Qualification'].isnull(), 'Applicant_Qualification'] = df_clean['Applicant_Qualification'].mode()[0]



#4 Converting Applicant and Manager Birthdate to Age instead of date (i.e. Age as of Jan 1 2009)
df_clean['Manager_DoB'] = pd.to_datetime(df_clean['Manager_DoB'])
df_clean['Manager_Age_When_Recruiting'] = (df_clean['Application_Receipt_Date'] - df_clean['Manager_DoB'])
df_clean['Manager_Age_When_Recruiting'] = df_clean['Manager_Age_When_Recruiting'].dt.days / 365
df_clean.loc[df_clean['Manager_Age_When_Recruiting'].isnull(), 'Manager_Age_When_Recruiting'] = df_clean['Manager_Age_When_Recruiting'].median()
df_clean['Manager_Age_When_Recruiting'] = df_clean['Manager_Age_When_Recruiting'].astype(int)


#5 Feature Engineering Application Recipt Date and Manager DOJ

df_clean['Application_Receipt_Year'] = df_clean['Application_Receipt_Date'].dt.strftime("%Y").astype(int)
df_clean['Application_Receipt_Month'] = df_clean['Application_Receipt_Date'].dt.strftime("%m").astype(int)
df_clean['Manager_DOJ'] = pd.to_datetime(df_clean['Manager_DOJ'])
df_clean['Manager_DOJ_Year'] = df_clean['Manager_DOJ'].dt.strftime("%Y").astype(int)
df_clean['Manager_DOJ_Month'] = df_clean['Manager_DOJ'].dt.strftime("%m").astype(int)

#6a Feature Engineering Applicant City PIN
df_clean['Applicant_City_Zone'] = 'Zone ' + df_clean['Applicant_City_PIN'].apply(str).str[0:1]  # Note that when Nan is converted to String, and the first letter is subset, it literally gets stored as n
df_clean['Applicant_City_Zone'].replace('Zone n', np.nan, inplace=True)
df_clean.loc[df_clean['Applicant_City_Zone'].isnull(), 'Applicant_City_Zone'] = df_clean['Applicant_City_Zone'].mode()[0]

#6b Feature Engineering Office PIN
df_clean['Office_PIN_Zone'] = 'Zone ' + df_clean['Office_PIN'].apply(str).str[0:1]  # Note that when Nan is converted to String, and the first letter is subset, it literally gets stored as n
df_clean['Office_PIN_Zone'].replace('Zone n', np.nan, inplace=True)
df_clean.loc[df_clean['Office_PIN_Zone'].isnull(), 'Office_PIN_Zone'] = df_clean['Office_PIN_Zone'].mode()[0]

#7 Feature Engineering the column: Manager_Years_In_Company
df_clean['Manager_Years_In_Company'] = df_clean['Application_Receipt_Date'] - df_clean['Manager_DOJ']
df_clean['Manager_Years_In_Company']  = df_clean['Manager_Years_In_Company'] .dt.days / 365

#7 Feature Engineering the column: Designation_Change
df_clean['Designation_Change'] = np.where(df_clean.Manager_Joining_Designation != df_clean.Manager_Current_Designation,1,0)

# Removing all unwanted columns
df_prep = df_clean[['Applicant_Gender', 'Applicant_Marital_Status', 'Applicant_Occupation', 'Applicant_Qualification', 'Manager_Joining_Designation', 'Manager_Current_Designation', 'Manager_Grade', 'Manager_Status', 'Manager_Gender',
					 'Manager_Num_Application', 'Manager_Num_Coded', 'Manager_Business', 'Manager_Num_Products', 'Manager_Business2', 'Manager_Num_Products2', 'Business_Sourced', 'Applicant_Age_At_Joining', 'Manager_Age_When_Recruiting',
					 'Application_Receipt_Year', 'Application_Receipt_Month', 'Applicant_City_Zone', 'Office_PIN_Zone', 'Manager_DOJ_Year', 'Manager_DOJ_Month', 'Manager_Years_In_Company', 'Designation_Change']]

df_final =
df_final = pd.get_dummies(df_final)



y_pred = logreg.predict(x_test)