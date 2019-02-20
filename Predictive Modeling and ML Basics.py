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



