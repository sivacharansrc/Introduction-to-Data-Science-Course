# INTRODUCTION TO PREDICTIVE MODELING

# Examples where predictive modeling can be useful
# Movie recommendations based on gender, age, sex, past movies etc
# Predicting the stock price etc
# Predictive modeling is using past data to predict future data

# Identifying factors responsible for Sales reduction - This is detective analysis rather than predictive modeling since
# no future forecast is involved

# TYPES OF PREDICTIVE MODELS
# Supervised Learning - A definite target variable is available from the historical data which can be used to predict
# future data Ex: House price prediction. The house price may be dependent on the locality, number of rooms, area etc

# Un-Supervised Learning - No specific target variables are available. Data are segmented based on the commonalities
# and the differences between each other. Ex: Google News grouping, segmentation of the customers etc

# Types of Supervised Learning - Target Variables are Continuous values (Regression) vs Discrete Values (Classification)
# UnSupervised Learning - Classic example is market segmentation.. Like segmenting customer base in to
# High Salary - High Spend, High Salary - Low Spend, Low Salary - High Spend, Low Salary - Low Spend

# STAGES OF PREDICTIVE MODELING

# Process of Predictive Modeling can be divided in to six stages
# 1 - Problem Definition
# 2 - Hypothesis Generation
# 3 - Data Extraction from all possible sources
# 4 - Data Exploration and Transformation
# 5 - Predictive Modeling
# 6 - Model Deployment / Implementation

# Problem Identification: Identifying the right problem, and formulating the problem mathematically
# Bad Problem Statement: Want to improve the profitability of credit card customers
# Reason: The above goal can be achieved in many different ways. Like, increasing the APR of the credit cards, or
# having different APR and benefits for different customer segments, or identifying customers with low default rate
# A problem statement should have a straight forward specific goal
# Good Problem Statement: Want to predict the default rate of customer (All the above defined possibilities rely
# on the customer default rate. Hence, predicting the default rate is more straight forward goal / prob. statement

# Hypothesis Generation: Listing down all possible variables, that might influence problem objective

# Let us analyze all the possible factors that might affect the above problem statement:
# Income: Higher Income people might have lower default rate
# Job Type: Person with a stable job might have a lower default rate
# Credit History: Previous repayment behavior can have an impact on future payment behavior
# Education: Educated people have better understanding of credit products, and may have lower default rate

# Hypothesis Generation should be done before looking at the data to avoid any bias

# Data Extraction:
# Collect data from as many sources as possible
# Also when we look at the data, we may come across any additional hypothesis as well
# The various data source may come from demographics of the customer, transaction history of the customer,
# payment history credit score from the bureau, and external competitive pricing information

# Data Exploration:
# Reading the data - pulling data from the data source to the work environment
# Variable Identification - We should identify the predictor and the target variable, and its data type
# Uni-variate Analysis - Analyze variables one by one by exploring bar plots, and histograms
# Bi-variate Analysis - Exploring the relation between two variables
# Missing Value Treatment - Identify variables with missing values, and imputing with mean, median, or mode etc.
# Outlier Treatment - Come up with methods to fix outliers
# Variable Transformation - Modify data to suit algorithm we wish to apply on data (log transformation on skewed data)

proj_dir = "C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course/"
path = proj_dir + "Data Files/Predictive Modeling and Machine Learning/data.csv"

import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('expand_frame_repr', False)

df = pd.read_csv(path)
df.shape
df.columns
df.head()

# Variable Identification:
# Process of identifying variables that are independent and dependent
# Identify the data type i.e. continuous vs categorical

# Dependent Variable: Variable that we are trying to predict (Ex: Survived in the above data set)
# Independent Variable: Variables that help in predicting dependent variable (Ex: sex, fare etc.)
# The dependent and independent variable can only be identified from the problem statement

# Category variables are stored as objects, whereas continuous variables are stored as int or float
# The variables can be identified using "dtypes" in pandas

df['PassengerId'].dtypes
df['Age'].dtypes
df['Name'].dtypes
df.dtypes

# Uni-variate analysis for continuous variables:
# What - Explore one variable at a time, and summarize the variable to discover insights, and anomalies etc.
#  Why - To identify the following
#           Central Tendency and Distribution - mean, median, standard deviation
#           Distribution of the variable - Symmetric, right skewed, or left skewed
#           Presence of missing values
#           Presence of Outliers

# Uni-variate analysis for Continuous Variable:
# Tabular Method: For analyzing mean, median, Standard Deviation, and missing values
# Graphical Method: Distribution of variables, and detecting outliers

# the "describe" function returns statistical summary for all continuous variables
df.describe()

# Graphical Method:
# One way of analyzing continuous data is using histogram

pd.DataFrame.hist(df)   # display histogram for all continuous variables in a pandas data frame

import matplotlib.pyplot as plt
plt.hist(df['Age'])

# Another way of analyzing the continuous data is using Box Plots

pd.DataFrame.boxplot(df)
plt.boxplot(df['Fare'], flierprops=dict(markerfacecolor='r', marker='D'))

# Uni-variate analysis for categorical variables:
# Some of the analysis that can be performed or everyone is interested in uni-variate analysis is:
# Count - absolute frequency of each category in a categorical variable
# Count% - proportion of different categories in a categorical variable expressed as %

proj_dir = "C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course/"
path = proj_dir + "Data Files/Predictive Modeling and Machine Learning/data.csv"

import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('expand_frame_repr', False)

df = pd.read_csv(path)

# Similar to continuous variables, we can perform either tabular methods or graphical methods
df['Sex'].value_counts()  # Frequency of the categorical variable
df['Sex'].value_counts() / df.shape[0]  # Frequency % of the categorical variable
df['Sex'].value_counts() / len(df['Sex'])  # Frequency % of the categorical variable

df['Sex'].value_counts().plot.bar()  # Bar plot for frequency
(df['Sex'].value_counts() / df.shape[0]).plot.bar()  # Bar plot for frequency %

# Bi-variate analysis
# When two variables are studied together to understand their relationship (or if any relationship exists)
# Bi-variate analysis helps in prediction

# Types of bi-variate analysis:
# Continuous - Continuous variables
import matplotlib.pyplot as plt

plt.scatter(df['Age'], df['Fare'])
df.plot.scatter('Age', 'Fare')


# Analyzing the figure there is not much information from the two variables

df['Age'].corr(df['Fare'])
df.corr()
# As figured from the scatter plot, there exists a very weak correlation between Age and Fare

# Categorical - Continuous Analysis
# Does the mean age of male different from that of female?

df.groupby('Sex')['Age'].mean()
df.groupby('Sex')['Age'].mean().plot.bar()

import scipy.stats as stats

(df['Age'][df['Sex'] == "male"]).std(ddof=1)
(df['Age'][df['Sex'] == "female"]).std(ddof=1)

stats.ttest_ind(df['Age'][df['Sex'] == "male"], df['Age'][df['Sex'] == "female"],  nan_policy='omit')
# The 2 sample t test suggest that there is a significant difference between the mean age of males from females

# Categorical - Categorical Analysis
# Does the gender have any relation on survival rates?

pd.crosstab(df['Sex'], df['Survived'])
stats.chi2_contingency(pd.crosstab(df['Sex'], df['Survived']))
# From the pvalue, it is evident that there is a significant difference between sex and survival rates
# Note that when we have an exclusive list of observed, then we can use the chisquare instead of the contigency function

# MISSING VALUES TREATMENT

# Types of Missing Values
#       Missing completely at random (MCAR) - missing values have no relation to the value iteself neither other values
#               For instance, in a dataset with two columns Age and IQ, the missing values in IQ column have
#               no relation with both Age or IQ

#       Missing at Random (MAR) - missing values have some related with other variables present in the data
#               For instance, if we have two columns Age, and IQ, then the IQ values are missing for any age < 55,
#               then this type of missing values are called MAR

#       Missing not at random (MNAR) - missing value have some sort of relation with the column itself
#               For example, in a data set with two columns Age, and IQ, if all values of IQ less than 100 are
#               missing, then they are called MNAR

# Identifying missing values
# describe - this is helpful for finding missing values only for continuous variable
# Isull - can be used for both continuous and categorical variable


# Treating Missing Values
# Imputation
#       Continuous - mean, median, mode,  and regression methods
#       Categorical - mode, classification model

# Deletion
#       Row wise deletion - Delete the entire row
#       Column wise deletion - Delete entire column

# Deletion treatment results in loss of data. So, imputation is preferred over deletion, unless required

proj_dir = "C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course/"
path = proj_dir + "Data Files/Predictive Modeling and Machine Learning/data.csv"

import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('expand_frame_repr', False)

df = pd.read_csv(path)

# Identify missing values of numerical variables
df.shape
df.describe()
len(df[df['Age'].isnull()])

# To identify all missing values in the data set
df.isnull().sum()
# From the above function, it is evident that we have missing values in age, cabin, and Embarked

# Drop any rows with missing values
df.dropna()
df.dropna().shape
df.dropna().isnull().sum()

# Drop only those rows where all values are missing
df.dropna(how='all').shape
# There are no rows where all values are missing

# Dropping any columns that have missing values
df.dropna(axis=1).shape   # 3 columns that had missing values were deleted

# Dropping columns that have all missing values
df.dropna(axis=1, how='all').shape   # There are no columns with all missing values

# Filling all na with zero
df.fillna(0)

# If the above action of filling na is to be done within the dataset, we use the argument inplace=T
# df.fillna(0, inplace=True)

# To impute a numerical variable with value other than zero
df['Age'].fillna(df['Age'].mean())

# OUTLIER TREATMENT
# Uni-variate Outliers: Analyze one variable for outliers - can be identified using box plot
# Bi-variate Outliers: Analyze two variables for outliers  - can be identified using scatter plot

# Identifying Outliers:
# Value less than Q1 - 1.5 IQR or greater than Q3 + 1.5 IQR
# IQR is the Inter Quartile Range which is Q3 - Q1

# Treating Outliers:
# Deleting Outliers
# Transforming and Binning Values
# Imputing Outliers similar to missing values
# Treat the outliers separately

import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('expand_frame_repr', 100)


proj_dir = "C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course/"
path = proj_dir + "Data Files/Predictive Modeling and Machine Learning/data.csv"
df = pd.read_csv(path)

pd.DataFrame.boxplot(df)
df['Age'].plot.box()
df['Fare'].plot.box()

df.plot.scatter('Age', 'Fare')
pd.scatter_matrix(df)

# Removing Outliers from the data set
# From the above scatter plot for Age and Fare, there are two data points that appear as outlier. One way to remove
# the outlier is to filter the data where the fare is greater than 300
df1 = df[df['Fare'] < 300]
df1.plot.scatter('Age', 'Fare')

# Replacing outliers in Age variable with the mean
df1.loc[df['Age'] > 65, 'Age'] = df1['Age'].mean()  # Outliers imputed with mean. They can also be done with median
# df1.loc[df['Age'] > 65, 'Age'] = df1['Age'].median()
# np.mean(df['Age']) also gives the mean.. np is from numpy package

df1['Age'].plot.box()

# Variable Transformation:

# Replace a variable with some function of that particular variable (i.e. replacing with log)
# Is a process in which we change the distribution or relationship of a variable with others

# Why used
# To change the scale of the variable. For instance, when we have 20 variables of which 17 are in Km, and 3 are in mile
# then the change in scale will create some issues. To bring the scale the same across the column, variable
# transformation are performed

# Transforming non linear relationships to linear relationships as linear relationships are easier to predict

# Create a symmetric distribution from a skewed distribution, as symmetric or normally distributed data
# are required for many model predictions etc

# Common methods of variable transformation

# Taking log of the variables reduces the right skewness of the variable
# Square Root - Used for right skewed variable, but applicable only for positive values
# Cube Root - Used for right skewed variable for both positive and negative variables
# Binning - convert continuous variable to categorical variable

import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('expand_frame_repr', 100)
import numpy as np
import matplotlib.pyplot as plt

proj_dir = "C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course/"
path = proj_dir + "Data Files/Predictive Modeling and Machine Learning/data.csv"
df = pd.read_csv(path)

df['Age'].plot.hist()

# Let us analyze the results of different transformations:

plt.subplot(221)
df['Age'].plot.hist()
plt.title = 'Age Histogram'
plt.subplot(222)
np.log(df['Age']).plot.hist()  # Taking log is completely changing the skewness to the left. Not very good
plt.title = 'Log transformation of Age Histogram'
plt.subplot(223)
np.sqrt(df['Age']).plot.hist()
plt.title = 'SQRT transformation of Age Histogram'
plt.subplot(224)
np.power(df['Age'], 1/3).plot.hist()
plt.title = 'Cube Root transformation of Age Histogram'
plt.show()

# Of all the above transformation, the sqrt gives better symmetry

# Transformation by Binning:
df['Age'].plot.hist()

# Let us consider from the above data set that people with age 0-15 are children, and the rest are adults
bins = [0, 15, 80]
group = ['child', 'adult']
df['Age Group'] = pd.cut(df['Age'], bins, labels=group)
df.head
# df['Age Group'].isnull().value_counts()
df['Age Group'].value_counts().plot(kind='bar')
df.groupby('Age Group').size().plot(kind='bar')

# BASICS OF MODEL BUILDING

# Three steps involved in model building:
#           Algorithm Selection
#           Training Model
#           Prediction or Scoring

# Chart for Algorithm Selection
# IF Dependent Variable exists, then Supervised Learning, else Unsupervised Learning
#           If Supervised Learning, Is dependent Variable continuous?
#                   If yes, then regression problem (Simple Linear Regression, Multiple Regression)
#                   If not, then classification problem (Logistic Regression, Decision Tree, Random Forest)
#           If Unsupervised, then Clustering problem (k-means, spectral clustering)


# LINEAR REGRESSIONS:
# r square is a measure of how good the model is
# r square = Sum of Square of Regression / Sum of Square of Total Variance (i.e. Sum((y hat - actual mean)^2) / Sum((y - actual mean)^2)
# r square can also be calculated from 1 - {Sum((y hat - y)^2) / Sum((y - actual mean)^2)}
# r square always lies between 0 and 1
# RMSE is one of the most popular metric for scoring regression models

# IMPLEMENTING LINEAR REGRESSION IN PYTHON

import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('expand_frame_repr', False)
import numpy as np
import matplotlib.pyplot as plt

proj_dir = "C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course/"
path = proj_dir + "Data Files/Predictive Modeling and Machine Learning/linear_regression_train.csv"
df = pd.read_csv(path)

df.dtypes
df.head()
df.shape


# In order for linear regression to run, all the variables should be numerical
# Categorical variables can be converted to numerical using the get_dummies function in pandas
# Creating dummies before split to avoid any conflicts between dummy variable creation

df_backup = df
df = pd.get_dummies(df,)


df_train = df[0:8000]
df_test = df[8000:]

x_train = df_train.drop('Item_Outlet_Sales', axis=1)
y_train = df_train['Item_Outlet_Sales']
x_test = df_test.drop('Item_Outlet_Sales', axis=1)
true_pred = df_test['Item_Outlet_Sales']

# Building a linear regression model
from sklearn.linear_model import LinearRegression
lreg = LinearRegression()


# For linear regressions to run properly, there should not be any missing values. For this case, let us fill NA with 0
x_train.fillna(0, inplace=True)
x_test.fillna(0, inplace=True)

lreg.fit(x_train, y_train)
pred = lreg.predict(x_test)

# Performance of the model:

lreg.score(x_test, true_pred)  # r square is 0.4029
lreg.score(x_train, y_train)  # r square is 0.6497
# From the above score test, r square has dropped significantly. The reason can be that the model  is
# over fitted for train data set, or the test data is not an actual representation of train

rmse_test = np.sqrt(np.mean(np.power((np.array(true_pred) - np.array(pred)), 2)))
rmse_train = np.sqrt(np.mean(np.power((np.array(y_train) - np.array(lreg.predict(x_train))), 2)))
# The RMSE results also very similar to r square. i.e. either there is over fitting or test is not representing train

pd.DataFrame.boxplot(df1)
pd.scatter_matrix(df1)

# UNDERSTANDING LOGISTIC REGRESSION:

# Logistic regression is basically a sigmoid function 1 / (1 + e^-yhat)
# Logistic regression is used for Yes / No classification. It determines whether a an outcome can occur or not, given the possibility of X
# Cost Function = -y (log(yhat)) - (1-y) log(1-yhat)

import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('expand_frame_repr', False)
import numpy as np
import matplotlib.pyplot as plt

proj_dir = "C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course/"
path = proj_dir + "Data Files/Predictive Modeling and Machine Learning/titanic.csv"
df = pd.read_csv(path)

df.head()
df['Survived'].value_counts()
df = pd.get_dummies(df)
df.fillna(0, inplace=True)
df.shape
train_data = df[0:699]
test_data = df[700:890]
x_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']
x_test = test_data.drop('Survived', axis=1)
y_test = test_data['Survived']

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg.fit(x_train, y_train)
predictions = logreg.predict(x_test)

logreg.score(x_test, y_test)
logreg.score(x_train, y_train)

# The accuracy has dropped significantly from 92 to 82. The major reason might be we are using the name of the passengers.
# Let us try rerun the logistic regression by removing the passenger name

df['Survived'].value_counts()
df['Cabin_Fixed'] = df['Cabin'].astype(str).str[0]  # To generalize the Cabin rather than having all seat number
df['Cabin_Fixed'][df['Cabin'].isnull()] = df['Cabin']
df.Fare.plot.hist(bins=20)  # To see if having a fare group can have an impact on the overall model
bins = [0, 20, 250, 500]
group = ['low', 'med', 'high']
df['Fare_Group'] = pd.cut(df['Fare'], bins, labels=group)
df.head()
pd.crosstab(df['Cabin_Fixed'], df['Fare_Group'])


df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df = pd.get_dummies(df)
df.fillna(0, inplace=True)
df.shape
train_data = df[0:699]
test_data = df[700:890]
x_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']
x_test = test_data.drop('Survived', axis=1)
y_test = test_data['Survived']

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg.fit(x_train, y_train)
predictions = logreg.predict(x_test)

logreg.score(x_test, y_test)  # 0.81
logreg.score(x_train, y_train)  # 0.79

accuracy_df = pd.DataFrame({'Actual_Values': y_test, 'Predicted_Values': predictions})
pd.crosstab(accuracy_df['Actual_Values'], accuracy_df['Predicted_Values'])

# Though the overall model did not improve, the test and train are more or less close which means that the model is now
# more generalized, rather than over fitting the training data

# Additional Reference: https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
# https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65

