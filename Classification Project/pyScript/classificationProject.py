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

# https://datahack.analyticsvidhya.com/contest/introduction-to-data-science-classification/

### Reading the data set

df = pd.read_csv("C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course/Classification Project/input/Train.csv")
df.head()

df.isnull().sum()
#1 From the above, it is evident that there are certain rows where all values related to manager is missing. All such rows should be removed from the data set
df.shape

#2 The ID column is not going to be helpful in predictions. So, the column can be totally removed from the data set

#3 Analyzing the missing values:
df_clean.isnull().sum()
# Columns with missing values: Applicant_City_PIN (80), Applicant_Gender (53), Applicant_BirthDate (59), Applicant_Marital_Status (59), Applicant_Occupation (1090), Applicant_Qualification (71)

# Analyzing whether applicant city PIN and Office PIN are same
df_clean['Applicant_City_PIN'].nunique()
df_clean['PIN_Match'] = np.where(df_clean['Office_PIN'] == df_clean['Applicant_City_PIN'], "Match", "No Match")
df_clean.PIN_Match.value_counts()
df_clean.drop('PIN_Match', axis=1, inplace=True)
# Most of the PIN did not match, and so dropping the column

# 53 missing values for both gender, and birth date. Checking whether both the variables having missing values in same rows, but this does not appear to be in the same row
(df_clean['Applicant_Gender'][df_clean['Applicant_BirthDate'].isnull()]).isnull().value_counts()

pd.crosstab(df_clean.Applicant_Gender, df_clean.Applicant_Occupation)

#3a Let us fill the Applicant Gender with mode

#3b Let us create a new column called Applicant_Age from the birthdate, fix outliers, and then fill missing values with median of the age
df_clean['Applicant_Age_At_Joining'].plot.box()
df_clean['Applicant_Age_At_Joining'].plot.hist()
df_clean['Business_Sourced'][df_clean['Applicant_Age_At_Joining']>57.9].value_counts()
#3b Let us not fix outliers for age, as the data is derieved from the birth date. Let us just impute the series with median

#3c Let us analyze for a threshold age for married vs unmarried

bins = [0, 10, 20, 25, 30, 40, 80]
group = ["0-10", "10-20", "20-25", "25-30", "30-40", "40-80"]
df_clean['Age_Group'] = pd.cut(df_clean['Applicant_Age'], bins, labels=group)

pd.crosstab(df_clean['Age_Group'], df_clean.Applicant_Marital_Status)
#3c Let us assume that age less than 25 are unmarried, and greater than 25 are married

#3d Analyzing Applicant Occupation
df_clean.Applicant_Occupation.value_counts()
df_clean.loc[df_clean.Applicant_Occupation.isnull(), 'Age_Group'].value_counts()

# After careful analysis, there is not any specific relation with other variables. Hence, let us create a new value for missing occupation as "Missing"

#3e Applicant_Qualification - Fixing the factors, and missing values
df_clean.Applicant_Qualification.value_counts()
# Looking at the different qualifications, some of the qualifications such as the associate, and professional marketing can be re-categorized under Others
pd.crosstab(df_clean.Age_Group, df_clean.Applicant_Qualification)
df_clean.Applicant_Qualification.isnull().value_counts()
# Let us fill the missing values with mode

#4 Convert Manager Birthdate to Age at the time of recruiting the applicant using the Application Receipt Date

#5 The Application Receipt date can be further broken down to month and year, and can then be removed

#6a & 6b: The Office PIN and Applicant City PIN can be grouped in to respective PIN Zones. Reference for Indian PIN Zone: https://en.wikipedia.org/wiki/Postal_Index_Number







#1 Remove all the rows where all variables related to manager is missing
df_clean = df[~df['Manager_DOJ'].isnull()]

#2 Removing the ID Column
df_clean = df_clean.drop('ID', axis=1)

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


# Analyzing the data set

# Checking whether the classes are balanced or not
df_prep.Business_Sourced.value_counts() / len(df_prep)
# There exists a class imbalance. We need to balance the data before we run the model

# Let us analyze the correlation with the different variables and the target variable

# Relation with application gender:
pd.crosstab(df_prep.Applicant_Gender, df_prep.Business_Sourced).plot.bar()
pd.crosstab(df_prep.Applicant_Marital_Status, df_prep.Business_Sourced).plot.bar()
pd.crosstab(df_prep.Applicant_Occupation, df_prep.Business_Sourced).plot.bar()
pd.crosstab(df_prep.Applicant_Qualification, df_prep.Business_Sourced).plot.bar()
pd.crosstab(df_prep.Manager_Grade, df_prep.Business_Sourced).plot.bar()
pd.crosstab(df_prep.Manager_Status, df_prep.Business_Sourced).plot.bar()
pd.crosstab(df_prep.Manager_Gender, df_prep.Business_Sourced).plot.bar()
pd.crosstab(df_prep.Office_PIN_Zone, df_prep.Business_Sourced).plot.bar()


pd.crosstab(df_prep.Applicant_Age_At_Joining, df_prep.Business_Sourced).plot.bar()
pd.crosstab(df_prep.Manager_Age_When_Recruiting, df_prep.Business_Sourced).plot.bar()
pd.crosstab(df_prep.Application_Receipt_Month, df_prep.Business_Sourced).plot.bar()
df_prep.Manager_Business.corr(df_prep.Business_Sourced)
pd.scatter_matrix(df_prep)

# Summary of columns that might have an impact on the target variable
keep = ['Applicant_Gender', 'Applicant_Occupation', 'Applicant_Qualification', 'Manager_Grade', 'Manager_Gender', 'Applicant_Age_At_Joining', 'Manager_Age_When_Recruiting', 'Designation_Change', 'Applicant_City_Zone', 'Office_PIN_Zone',
 'Manager_Years_In_Company', 'Manager_Num_Application', 'Manager_Num_Coded', 'Manager_Business', 'Manager_Num_Products', 'Manager_Business2', 'Manager_Num_Products2', 'Business_Sourced']

df_final = df_prep[keep]
df_final = pd.get_dummies(df_final)

# Reference: https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

# Creating the Train and Test Data

x = df_final.drop('Business_Sourced', axis=1)
y = df_final['Business_Sourced']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, stratify=y, test_size=0.3)

# Oversampling the train data to balance the imbalanced data
from imblearn.over_sampling import SMOTE

os = SMOTE(random_state=0)

columns = x_train.columns
x_train,y_train =os.fit_sample(x_train, y_train)
x_train = pd.DataFrame(data=x_train,columns=columns )
y_train= pd.DataFrame(data=y_train,columns=['y'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(x_train))
print("Number of no subscription in oversampled data",len(y_train[y_train['y']==0]))
print("Number of subscription",len(y_train[y_train['y']==1]))
print("Proportion of no subscription data in oversampled data is ",len(y_train[y_train['y']==0])/len(x_train))
print("Proportion of subscription data in oversampled data is ",len(y_train[y_train['y']==1])/len(x_train))

# Initiate a logistic regression object
log_reg = LogisticRegression()

rfe = RFE(log_reg, 15)
rfe = rfe.fit(x_train, y_train.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

col_names = x_train.columns
rfe_output = rfe.ranking_
colToSelect = pd.DataFrame({'cols': col_names, 'rfe': rfe_output})
colToSelect = colToSelect.loc[colToSelect.rfe == 1, 'cols']
colToSelect = pd.Series.tolist(colToSelect)
x_train = x_train[colToSelect]

# Running Stats Model on the Selected Features

logistic_model = sm.Logit(y_train, x_train)
result = logistic_model.fit()
print(result.summary2())

stat_sig_col = ['Applicant_Occupation_Others', 'Applicant_Occupation_Salaried','Applicant_Occupation_Student', 'Applicant_Qualification_Class X','Applicant_Qualification_Class XII', 'Applicant_City_Zone_Zone 1',
				'Applicant_City_Zone_Zone 2', 'Office_PIN_Zone_Zone 3']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, stratify=y, test_size=0.2)

x_train = x_train[stat_sig_col]
x_test = x_test[stat_sig_col]

logreg = LogisticRegression()
logreg.fit(x_train, y_train)

# Predicting the results

y_pred = logreg.predict(x_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# ROC Curve

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
plt.show()