import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('expand_frame_repr', False)
import numpy as np


# Reading Training Data Set:
df_train = pd.read_csv("C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course/Classification Project/input/Train.csv")
df_test = pd.read_csv("C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course/Classification Project/input/Test.csv")
df_test['Business_Sourced'] = np.nan
df = pd.concat([df_train, df_test], axis=0) # 5045 Test data Set rows

colsToKeep = ['ID', 'Office_PIN', 'Application_Receipt_Date', 'Applicant_City_PIN', 'Applicant_Gender', 'Applicant_BirthDate', 'Applicant_Marital_Status', 'Applicant_Occupation', 'Applicant_Qualification', 'Manager_DOJ', 'Manager_Joining_Designation', 'Manager_Current_Designation',
			  'Manager_Status', 'Manager_Gender', 'Manager_DoB', 'Manager_Num_Application', 'Manager_Num_Coded', 'Business_Sourced']

df = df.loc[:, colsToKeep]
# (df.isnull().sum() / len(df))*100 - Most number of missing data is found in Applicant Occupation which is 15%. Since, this is well below the threshold of 30% we can still go ahead and impute all columns for our modelling

# FEATURE ENGINEERING APPLICANT ZONE AND OFFICE ZONE
df.loc[df['Applicant_City_PIN'].isnull(), 'Applicant_City_PIN'] = df['Applicant_City_PIN'].mode()[0]
df['Applicant_City_Zone'] = 'Zone ' + df['Applicant_City_PIN'].apply(str).str[0:1]  # Note that when Nan is converted to String, and the first letter is subset, it literally gets stored as n
df['Applicant_City_Zone'].replace('Zone n', np.nan, inplace=True)

df.loc[df['Office_PIN'].isnull(), 'Office_PIN'] = df['Office_PIN'].mode()[0]
df['Office_Zone'] = 'Zone ' + df['Office_PIN'].apply(str).str[0:1]  # Note that when Nan is converted to String, and the first letter is subset, it literally gets stored as n
df['Office_Zone'].replace('Zone n', np.nan, inplace=True)


# FEATURE ENGINEERING APPLICANT RECEIPT MONTH AND APPLICANT RECEIPT DAY - EXTRACTING MONTH AND DAY FROM APPLICANT RECEIPT DATE
df.loc[df['Application_Receipt_Date'].isnull(), 'Application_Receipt_Date'] = df['Application_Receipt_Date'].mode()[0]
df['Application_Receipt_Date'] = pd.to_datetime(df['Application_Receipt_Date'])
df['Application_Receipt_Month'] = df['Application_Receipt_Date'].dt.strftime("%m").astype(int)
df['Application_Receipt_Day'] = df['Application_Receipt_Date'].dt.weekday


# FEATURE ENGINEERING APPLICANT AGE AND MANAGER AGE FROM THE RESPECTIVE DOB AND APPLICATION RECEIPT DATE
df['Applicant_Age'] = df['Applicant_BirthDate']
df.loc[df['Applicant_Age'].isnull(), 'Applicant_Age'] = df['Applicant_Age'].mode()[0]  # Temporarily filling na's with mode to overcome any date conversion errors
df['Applicant_Age'] = pd.to_datetime(df['Applicant_Age'])
df['Applicant_Age'] = (df['Application_Receipt_Date'] - df['Applicant_Age'])
df['Applicant_Age'] = df['Applicant_Age'].dt.days / 365
df.loc[df['Applicant_BirthDate'].isnull(), 'Applicant_Age'] = np.nan
df.loc[df['Applicant_Age'].isnull(), 'Applicant_Age'] = df['Applicant_Age'].mean()

df['Manager_Age'] = df['Manager_DoB']
df.loc[df['Manager_Age'].isnull(), 'Manager_Age'] = df['Manager_Age'].mode()[0]  # Temporarily filling na's with mode to overcome any date conversion errors
df['Manager_Age'] = pd.to_datetime(df['Manager_Age'])
df['Manager_Age'] = (df['Application_Receipt_Date'] - df['Manager_Age'])
df['Manager_Age'] = df['Manager_Age'].dt.days / 365
df.loc[df['Manager_DoB'].isnull(), 'Manager_Age'] = np.nan
df.loc[df['Manager_Age'].isnull(), 'Manager_Age'] = df['Manager_Age'].mean()


# IMPUTING APPLICATION GENDER AND MANAGER GENDER WITH MODE
df.loc[df['Applicant_Gender'].isnull(), 'Applicant_Gender'] = df['Applicant_Gender'].mode()[0]
df.loc[df['Manager_Gender'].isnull(), 'Manager_Gender'] = df['Manager_Gender'].mode()[0]

# IMPUTING APPLICANT MARITAL STATUS -  < 25 as S and > 25 as M

df.loc[(df['Applicant_Age'] < 25) & (df['Applicant_Marital_Status'].isnull()), 'Applicant_Marital_Status'] = "S"
df.loc[(df['Applicant_Age'] >= 25) & (df['Applicant_Marital_Status'].isnull()), 'Applicant_Marital_Status'] = "M"

# IMPUTING APPLICANT OCCUPATION - MISSING VALUES WITH "MISSING"
df.loc[df['Applicant_Occupation'].isnull(), 'Applicant_Occupation'] = "Missing"

# IMPUTING APPLICANT QUALIFICATION - RE-CATEGORIZING
df.loc[(~df['Applicant_Qualification'].isin(['Class XII', 'Class X', 'Graduate', 'Masters of Business Administration'])) & (~df['Applicant_Qualification'].isnull()), 'Applicant_Qualification'] = "Others"
df.loc[df['Applicant_Qualification'].isnull(), 'Applicant_Qualification'] = df['Applicant_Qualification'].mode()[0]


# FEATURE ENGINEERING MANAGER EXPERIENCE IN COMPANY
df['Manager_Experience_In_Company'] = df['Manager_DOJ']
df.loc[df['Manager_Experience_In_Company'].isnull(), 'Manager_Experience_In_Company'] = df['Manager_Experience_In_Company'].mode()[0]
df['Manager_DOJ'] = pd.to_datetime(df['Manager_DOJ'])
df['Manager_Experience_In_Company'] = df['Application_Receipt_Date'] - df['Manager_DOJ']
df['Manager_Experience_In_Company']  = df['Manager_Experience_In_Company'] .dt.days / 365
df['Manager_Experience_In_Company']  = np.where(df['Manager_Experience_In_Company'] < 0, 0, df['Manager_Experience_In_Company'])
df.loc[df['Manager_DOJ'].isnull(), 'Manager_Experience_In_Company'] = np.nan
df.loc[df['Manager_DOJ'].isnull(), 'Manager_Experience_In_Company'] = df['Manager_Experience_In_Company'].median()


# FEATURE ENGINEERING DESIGNATION CHANGE (~~~~ TRY IF RETAINING MISSING VALUES AND CONSIDERING THEM NO DESIGNATION CHANGE IMPROVES THE ACCURACY OF THE MODEL ~~~~~~)
df.loc[df['Manager_Joining_Designation'].isnull(), 'Manager_Joining_Designation'] = df['Manager_Joining_Designation'].mode()
df.loc[df['Manager_Current_Designation'].isnull(), 'Manager_Current_Designation'] = df['Manager_Current_Designation'].mode()
df['Designation_Change'] = np.where(df.Manager_Joining_Designation != df.Manager_Current_Designation,1,0)

# FEATURE ENGINEERING MANAGER'S CAPABILITY TO CONVERT THE SOURCED APPLICATION TO RECRUITMENT
df.loc[df['Manager_Num_Coded'].isnull(), 'Manager_Num_Coded'] = 0
df.loc[df['Manager_Num_Application'].isnull(), 'Manager_Num_Application'] = 0
df['Application_Conversion_Ratio'] = np.where((df.Manager_Num_Coded == 0) | (df.Manager_Num_Application == 0), 0, df.Manager_Num_Coded / df.Manager_Num_Application)
df.loc[df['Application_Conversion_Ratio'] > 1, 'Application_Conversion_Ratio'] = 1

# IMPUTING MANAGER STATUS WITH MISSING
df.loc[df['Manager_Status'].isnull(), 'Manager_Status'] = "Missing"

# CONVERTING BUSINESS SOURCED TO INT
df.loc[df['Business_Sourced'].isnull(), 'Business_Sourced'] = 0
df['Business_Sourced'] = df['Business_Sourced'].astype(int)

# REMOVING UNNECESSARY COLUMNS
df.drop(['Application_Receipt_Date', 'Manager_DoB', 'Applicant_BirthDate', 'Applicant_City_PIN', 'Office_PIN', 'Manager_DOJ', 'Manager_Joining_Designation', 'Manager_Current_Designation', 'Manager_Num_Coded', 'Manager_Num_Application', 'Applicant_City_Zone', 'Office_Zone'], axis=1, inplace=True)

# ENCODING CATEGORICAL VARIABLES

one_hot_encoding = ['Applicant_Gender', 'Applicant_Marital_Status', 'Manager_Status', 'Manager_Gender', 'Applicant_Occupation', 'Applicant_Qualification']
df_one_hot = df.loc[:, one_hot_encoding]
df.drop(one_hot_encoding, axis=1, inplace=True)

# PERFORMING ONE HOT ENCODING
df_one_hot = pd.get_dummies(df_one_hot, drop_first=True)

# ALL TRANSFORMATIONS IN ONE HOT ENCODING AS THIS GIVES BETTER ACCURACY

#label_encoding = ['Applicant_Qualification']
#df_label = df.loc[:, label_encoding]
#df.drop(label_encoding, axis=1, inplace=True)

# PERFORMING LABEL ENCODING
#from sklearn.preprocessing import LabelEncoder
#label_encode = LabelEncoder()
#df_label = df_label.apply(lambda col: label_encode.fit_transform(col))

# CONCAT ALL THE DATA FRAMES
df = pd.concat([df, df_one_hot], axis=1)
# df = pd.concat([df, df_one_hot, df_label], axis=1)

# SEGREGATING THE TRAIN AND THE TEST DATA

train = df[0:len(df_train)]
test = df[len(df_train):len(df)]

# TRAINING FOR LOGISTIC REGRESSION MODEL

y = train['Business_Sourced']
x = train.drop(['Business_Sourced','ID'], axis=1)

# SPLITTING THE DATA IN TO TRAIN AND VALIDATION
from sklearn.model_selection import train_test_split

x_train, x_validation, y_train, y_validation = train_test_split(x,y, test_size=0.3, random_state=1, stratify=y)

# NO SCALING TO BE PERFORMED AS THERE IS NO EFFECT ON THE ACCURACY
# SCALING DATA BEFORE BUILDING MODEL
# SCALE USING MIN MAX SCALER
#from sklearn.preprocessing import MinMaxScaler

#scaled_data = MinMaxScaler(feature_range=(0,1))
#scaled_data.fit(x_train)

#columns = x_train.columns
#x_train = scaled_data.transform(x_train)
#x_validation = scaled_data.transform(x_validation)
#x_train = pd.DataFrame(data=x_train,columns=columns)
#x_validation = pd.DataFrame(data=x_validation,columns=columns)

# SCALE USING STANDARD SCALER
#from sklearn.preprocessing import StandardScaler

#scaled_data = StandardScaler()
#scaled_data.fit(x_train)

#columns = x_train.columns
#x_train = scaled_data.transform(x_train)
#x_validation = scaled_data.transform(x_validation)
#x_train = pd.DataFrame(data=x_train,columns=columns)
#x_validation = pd.DataFrame(data=x_validation,columns=columns)

# Oversampling the train data to balance the imbalanced data
from imblearn.over_sampling import SMOTE

os = SMOTE(random_state=0)

columns = x_train.columns
x_train,y_train =os.fit_sample(x_train, y_train)
x_train = pd.DataFrame(data=x_train,columns=columns )
#y_train= pd.DataFrame(data=y_train,columns=['Business_Sourced'])


# BUILDING A LOGISTIC REGRESSION MODEL
from sklearn.linear_model import LogisticRegression

#logistic_regression = LogisticRegression(solver='lbfgs', max_iter=1000) # Default solver was warn, but in future versions lbfgs would be the new default. The lbfgs required more iterations due to convergence issue. Convergence issue can be resolved by scaling as well
logistic_regression = LogisticRegression(solver='liblinear', penalty='l2', max_iter=150) # Default solver was warn, but in future versions lbfgs would be the new default. The lbfgs required more iterations due to convergence issue. Convergence issue can be resolved by scaling as well
logistic_regression.fit(x_train, y_train)
logistic_regression.score(x_train, y_train)
logistic_regression.score(x_validation, y_validation)
predictions = logistic_regression.predict(x_validation)

# CONFUSION MATRIX AND CLASSIFICATION REPORT
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#y_validation = y_validation.values

conf_matrix = confusion_matrix(y_validation, predictions)
class_report = classification_report(y_validation.tolist(), predictions.tolist())

print(conf_matrix)
print(class_report)

# PLOTTING THE ROC CURVE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pylab as plt

logit_roc_auc = roc_auc_score(y_validation, predictions)
fpr, tpr, thresholds = roc_curve(y_validation, logistic_regression.predict_proba(x_validation)[:,1])
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

# RUNNING THE MODEL ON TEST DATA

ID = test['ID']
test.drop(['ID', 'Business_Sourced'], axis=1, inplace=True)

test = scaled_data.fit_transform(test)

predictions = logistic_regression.predict(test)
output = pd.DataFrame({'ID': ID, 'Business_Sourced': predictions})

output.to_csv("C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course/Classification Project/output/Submission 13 - Defaulting to liblinear solver with 150 iterations.csv", index=False, header=True)

# UNDERSTANDING THE IMPORTANCE OF THE FEATURES

import statsmodels.api as sm

logistic_model = sm.Logit(y_train, x_train)
result = logistic_model.fit()
print(result.summary2())

# RUNNING A BASIC DECISION TREE ALGORITHM
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=0.6, max_features=0.6, max_leaf_nodes=8, criterion='gini', min_samples_split=6)
run_decision_tree()

def run_decision_tree():
	clf.fit(x_train, y_train)
	clf.score(x_train, y_train)
	predictions = clf.predict(x_validation)
	train_predictions = clf.predict(x_train)
	clf.score(x_validation, y_validation)

	conf_matrix = confusion_matrix(y_validation, predictions)
	class_report = classification_report(y_validation.tolist(), predictions.tolist())

	print(conf_matrix)
	print(class_report)

	from sklearn.metrics import roc_curve, auc

	false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_predictions)
	roc_auc = auc(false_positive_rate, true_positive_rate)
	print(roc_auc)


	false_positive_rate, true_positive_rate, thresholds = roc_curve(y_validation, predictions)
	roc_auc = auc(false_positive_rate, true_positive_rate)
	print(roc_auc)

# PREDICTING THE TEST DATA

ID = test['ID']
test.drop(['ID', 'Business_Sourced'], axis=1, inplace=True)


predictions = clf.predict(test)
output = pd.DataFrame({'ID': ID, 'Business_Sourced': predictions})

output.to_csv("C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course/Classification Project/output/Submission 14 - Basic Decision Tree.csv", index=False, header=True)




# SUBMISSION 7
# df.drop(['Application_Receipt_Date', 'Manager_DoB', 'Applicant_BirthDate', 'Applicant_City_PIN', 'Office_PIN', 'Manager_DOJ', 'Manager_Joining_Designation', 'Manager_Current_Designation', 'Manager_Num_Coded', 'Manager_Num_Application', 'Applicant_City_Zone', 'Office_Zone'], axis=1, inplace=True)