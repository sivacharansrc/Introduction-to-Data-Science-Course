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

df = pd.read_csv("C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course/Classification Project/input/Train.csv")
df.dropna(axis=0, inplace=True)

df.boxplot(column='Manager_Business', by='Manager_Current_Designation')

# Feature Engineering

#3a Filling Applicant_Gender missing values with mode
df.loc[df['Applicant_Gender'].isnull(), 'Applicant_Gender'] = df['Applicant_Gender'].mode()[0]

#3b Creating Applicant_Age from the birth date, and imputing missing data with median
df['Applicant_BirthDate'] = pd.to_datetime(df['Applicant_BirthDate'])
df['Application_Receipt_Date'] = pd.to_datetime(df['Application_Receipt_Date'])
df['Applicant_Age_At_Joining'] = (df['Application_Receipt_Date'] - df['Applicant_BirthDate'])
df['Applicant_Age_At_Joining'] = df['Applicant_Age_At_Joining'].dt.days / 365
df.loc[df['Applicant_Age_At_Joining'].isnull(), 'Applicant_Age_At_Joining'] = df['Applicant_Age_At_Joining'].median()
df['Applicant_Age_At_Joining'] = df['Applicant_Age_At_Joining'].astype(int)

#3c Filling Applicant_Marital_Status < 25 as S and > 25 as M

df.loc[(df['Applicant_Age_At_Joining'] < 25) & (df['Applicant_Marital_Status'].isnull()), 'Applicant_Marital_Status'] = "S"
df.loc[(df['Applicant_Age_At_Joining'] >= 25) & (df['Applicant_Marital_Status'].isnull()), 'Applicant_Marital_Status'] = "M"

#3d Filling Applicant_Occupation missing values as missing

df.loc[df['Applicant_Occupation'].isnull(), 'Applicant_Occupation'] = "Missing"

#3e Fixing the Applicant_Qualification - Recategorizing,

df.loc[(~df['Applicant_Qualification'].isin(['Class XII', 'Class X', 'Graduate', 'Masters of Business Administration'])) & (~df['Applicant_Qualification'].isnull()), 'Applicant_Qualification'] = "Others"
df.loc[df['Applicant_Qualification'].isnull(), 'Applicant_Qualification'] = df['Applicant_Qualification'].mode()[0]



#4 Converting Applicant and Manager Birthdate to Age instead of date (i.e. Age as of Jan 1 2009)
df['Manager_DoB'] = pd.to_datetime(df['Manager_DoB'])
df['Manager_Age_When_Recruiting'] = (df['Application_Receipt_Date'] - df['Manager_DoB'])
df['Manager_Age_When_Recruiting'] = df['Manager_Age_When_Recruiting'].dt.days / 365
df.loc[df['Manager_Age_When_Recruiting'].isnull(), 'Manager_Age_When_Recruiting'] = df['Manager_Age_When_Recruiting'].median()
df['Manager_Age_When_Recruiting'] = df['Manager_Age_When_Recruiting'].astype(int)


#5 Feature Engineering Application Recipt Date and Manager DOJ

df['Application_Receipt_Year'] = df['Application_Receipt_Date'].dt.strftime("%Y").astype(int)
df['Application_Receipt_Month'] = df['Application_Receipt_Date'].dt.strftime("%m").astype(int)
df['Manager_DOJ'] = pd.to_datetime(df['Manager_DOJ'])
df['Manager_DOJ_Year'] = df['Manager_DOJ'].dt.strftime("%Y").astype(int)
df['Manager_DOJ_Month'] = df['Manager_DOJ'].dt.strftime("%m").astype(int)

#6a Feature Engineering Applicant City PIN
df['Applicant_City_Zone'] = 'Zone ' + df['Applicant_City_PIN'].apply(str).str[0:1]  # Note that when Nan is converted to String, and the first letter is subset, it literally gets stored as n
df['Applicant_City_Zone'].replace('Zone n', np.nan, inplace=True)
df.loc[df['Applicant_City_Zone'].isnull(), 'Applicant_City_Zone'] = df['Applicant_City_Zone'].mode()[0]

#6b Feature Engineering Office PIN
df['Office_PIN_Zone'] = 'Zone ' + df['Office_PIN'].apply(str).str[0:1]  # Note that when Nan is converted to String, and the first letter is subset, it literally gets stored as n
df['Office_PIN_Zone'].replace('Zone n', np.nan, inplace=True)
df.loc[df['Office_PIN_Zone'].isnull(), 'Office_PIN_Zone'] = df['Office_PIN_Zone'].mode()[0]

#7 Feature Engineering the column: Manager_Years_In_Company
df['Manager_Years_In_Company'] = df['Application_Receipt_Date'] - df['Manager_DOJ']
df['Manager_Years_In_Company']  = df['Manager_Years_In_Company'] .dt.days / 365

#7 Feature Engineering the column: Designation_Change
df['Designation_Change'] = np.where(df.Manager_Joining_Designation != df.Manager_Current_Designation,1,0)