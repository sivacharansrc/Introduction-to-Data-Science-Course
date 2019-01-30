""" INTRODUCTION TO DESCRIPTIVE STATISTICS """

import pandas as pd
pd.set_option('display.max_column', 100)
pd.set_option('expand_frame_repr', False)
project_dir = "C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course"
path = project_dir+"/Data Files/Understanding Statistics/mode.csv"
df = pd.read_csv(path)
df.shape
df.head()

""" MEASURES OF CENTRAL TENDENCY """

df_mode = df['Subject'].mode()
print(df_mode)
# Mode is generally calculated for categorical variables to understand the frequency of occurence
# of the variable. A series can have single mode, or bi-mode (two variables occur at the same
# maximum frequency)

path = project_dir+"/Data Files/Understanding Statistics/mean.csv"
df = pd.read_csv(path)

marks_mean = df['Overall Marks'].mean()
print(marks_mean)

marks_mode = df.iloc[:, 1].mode()
marks_mode

path1 = project_dir+"/Data Files/Understanding Statistics/mean_robust.csv"
df1 = pd.read_csv(path1)

df1.tail()
df1['Overall Marks'].mean()
df1['Overall Marks'].mode()

# The second dataset had some outlier which is skewing the mean data. Presence of an outlier, or a extreme value
# can greatly skew the mean data. Hence, mean is not a robust metric

primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
sum(primes) / len(primes)  # Note that .mean is applicable only to dataframe column, but not list

# Outliers in datasets

# A value that lies in an abnormal distance from the rest of the values is an outlier

# Some common reasons for outliers: typo, measurement error, intentional error, legit outliers

withoutOutlier = {'withoutOutlier': [1, 2, 3, 4, 5, 6]}
withOutlier = {'withOutlier': [1, 2, 3, 4, 5, 6, 99]}

withOutlier = pd.DataFrame(data=withOutlier)
withoutOutlier = pd.DataFrame(data=withoutOutlier)

withOutlier['withOutlier'].mean()
withoutOutlier['withoutOutlier'].mean()

# Median of a dataset

path = project_dir+"/Data Files/Understanding Statistics/median.csv"
df = pd.read_csv(path)

df.head()
df['Overall Marks'].median()

#  Calculating quantiles: Quantiles are 1/4 th of a Quartile

df['Overall Marks'].quantile(0.25)
df['Overall Marks'].quantile(0.5)
df['Overall Marks'].quantile(0.75)
df['Overall Marks'].quantile(1)

# When there are no outliers, mean is the best way to measure the central tendency of data. However, if the data
# has outliers, median is the better way to measure the central tendency

# EXERCISE PROBLEMS
import pandas as pd
df = pd.DataFrame(data={'med_data': range(0, 20)})
df['med_data'].median()

""" SPREAD OF DATA """

# The central value is not always sufficient to describe the data. So, it is important to know the spread of
# the data in the dataset. i.e. How far is each data near or close to each other

df = pd.DataFrame(data={'range_data': range(1,21)})

# Range is another measure to calculate the spread of the data. i.e. Max Value - Min Value. However, this
# measure again does not describe the actual summary of the data. For instance, let us assume a dataset
# containing values between 1 and 10. The range of the dataset is 9. However, if there is a single record in
# the dataset which is an outlier, (say 100 instead of 10) the range becomes 99, which is not an actual representation
# of the data. In such cases, the IQR or the Inter Quartlie Range provides a better summary of the data
# IQR is calculated by 3rd quartile range - 1st quartile range.

project_dir = "C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course"
path = project_dir+"/Data Files/Understanding Statistics/Spread of Data.csv"
df = pd.read_csv(path)

df_range = df['Overall Marks'].max() - df['Overall Marks'].min()
df_IQR = df['Overall Marks'].quantile(0.75) - df['Overall Marks'].quantile(0.25)
df_IQR

# EXERCISE PROBLEMS

df = pd.DataFrame(data={'data': [37, 42, 48, 51, 52, 53, 54, 54, 55]})
df['data'].quantile(0.75)

""" VARIANCE OF THE DATA """

# Let us use the same data set that we used for IQR
df = pd.read_csv(path)

marks_mean = df['Overall Marks'].mean()
marks_dist_from_mean = df['Overall Marks'] - marks_mean
squared_dist = marks_dist_from_mean ** 2
mean_squared_dist = squared_dist.mean()
mean_squared_dist

df['Overall Marks'].var(ddof=0)  # Calculating Variance using the var function

# EXERCISE PROBLEMS
df = pd.DataFrame(data={'data': [7, 6, 8, 4, 2, 7, 6, 7, 6, 5]})['data'].var(ddof=0)  # Always use ddof (deg. of freedom) 0 when calculating variance. default value is 1
df

""" STANDARD DEVIATION OF DATA """

# Let us use the same data set that we used for IQR
df = pd.read_csv(path)

variance = df['Overall Marks'].var(ddof=0)
stdDev = variance ** 0.5
stdDev

stdDev = df['Overall Marks'].std(ddof=0)  # Standard Deviation function
stdDev

# EXERCISE PROBLEMS

df = pd.DataFrame(data={'data': [180, 313, 101, 255, 202, 198, 109, 183, 181, 113, 171, 165, 318, 145, 131, 145, 226, 113, 268, 108]})
len(df[(df['data'] < (df['data'].std(ddof=0) + df['data'].mean())) & (df['data'] > (df['data'].mean() - df['data'].std(ddof=0)))])  # Number of values lying between 1 SD

""" DATA DISTRIBUTION """

# FREQUENCY TABLES

df = pd.read_csv(path)
df['Subject'].value_counts()

# HISTOGRAMS
import matplotlib.pyplot as plt  # Library for plotting and visualizations
%matplotlib inline  # Does not work in pycharm, but works in jupyter notebooks.
# This function basically helps to display within the page the code resides rather than opening a new page
%matplotlib notebook  # works only in jupyter notebook

path = project_dir+"/Data Files/Understanding Statistics/Histogram.csv"
df = pd.read_csv(path)

plt.hist(x='Overall Marks', data=df, bins=10)
plt.hist(x='Overall Marks', data=df, bins=20)

# INTRODUCTION TO PROBABILITY

# Probability distribution of random discrete variable is called Probability Mass Function
# Probability distribution of a random continuous variable is called Probability Density Function

# CENTRAL LIMIT THEOREM
# If we take multiple random samples from a population, and plot the means of each of the sample, the
# distribution follows a bell curve which is called the normal distribution when we have taken enough sample
# from the population
# The theorem also states that the mean of the means will be approximately equal to the mean of sample means

# PROPERTIES OF NORMAL DISTRIBUTION:
# Area under the Probability Density Function gives the probability of the random variable to be in that range
# Multiple Samples of equal size from a Population data, and plotting the means gives a normal distributed curve
# There is a large possibility that the means to be around the actual mean of the data, than to be farther away
# Normal distributions of higher SD are flatter as compared to those for lower SD

# USING NORMAL CURVE FOR CALCULATIONS

# 68% of data lies within 1 SD and 95% of data lies within 2 SD
# In a normal distribution, the value of mode will be equal to 0 (since the distribution is from -1 to 1)

# Z SCORES

# The average IQ is 100, with a SD of 15. What percentage of population would you expect to have an IQ more
# than 120
# 100 + Z15 = 120
# Z = (120 - 100) / 15
# Z = 1.33
# From Z Score table, value of 1.33 equals 0.9082
# i.e Around 91% constitute up to 120 IQ. Therefore around 9% have more than 120% IQ

# PROBLEM QUESTIONS
# Standard normal probability distribution has mean equal to 40, whereas value of random variable x is 80
# and z-statistic is equal to 2 then standard deviation of standard normal probability distribution is

# 40 + 2 SD = 80
# sd = 20

# What's the z-score when the population mean is 40, standard deviation is 7, and x=30?
# 40 + Z7 = 30
# Z = -10 / 7
# Z = - 1.42

""" INTRODUCTION TO INFERENTIAL STATISTICS """

# Make inferences about the population from the sample
# Concluding whether a sample is significantly different from the population
# Hypothesis testing in general

# TERMINOLOGIES

# Statistic - Single measure of some attribute of a sample
# Population Statistic - The Statistic of the entire population in context
# Sample Statistic - The statistic of a group taken from a population
# Standard Deviation - The amount of variation in the population data. It is given by sigma

# Sampling Distribution:
    # We have salaries of all data scientists in India
    # We take random samples of 200 people and calculate and plot their means
    # In general Sampling Distribution is the normal curve distribution obtained by plotting the sample means

# The Central Limit Theorem:
    # While plotting a sampling distribution, the mean of the sample means is equal to the population mean
    # The sampling distribution approaches a normal mean
    # The CLT holds true irrespective of the type of distribution of the population
    # Greater the sample size, greater the accuracy of the population mean determined from the sample mean

# CONFIDENTIAL INTERVAL AND MARGIN OF ERROR

# The confidence interval is a type of interval estimate from the sampling distribution which gives a range of
# values in which the population statistic may lie
# A 95% CI means that interval estimates will contain the population statistic 95% of the time

# Calculate the 95% confidence interval for a sample mean of 40 and sample SD of 40 with sample size 100

# Confidence Interval = Mean +- Z * sigma / sqrt(N) # sigma is otherwise the SD
# From the Z table, the Z value of Z Score 95% is +- 1.96
# CI = [40 - (1.96 * 40 / 10)], [40 + (1.96 * 40 / 10)]
# CI = [32.16], [47.84]

# MARGIN OF ERROR

# Margin of Error is the half of Confidence Interval
# Sampling error by the person who collected the samples
# If the sample mean lies in the margin of error range, then it might be possible that its actual value is
# equal to the population mean, and the difference is occurring by chance
# Anything outside the margin of error is considered statistically significant
# Margin of error is on the either side of the mean. It can be both positive and negative

# Hypothesis Testing:


# Null Hypothesis: The sample statistic is equal to the population statistic, or that the intervention does not being
# any difference to the sample

# Alternate Hypothesis: This basically negates the Null Hypothesis, or states that the intervention does have an impact
# over the sample, and that the sample statistic is significantly different from the population statistic


# Problem Statement: Assume that a class is performing poor on all subjects. The teacher thinks that the introduction
# of music while teaching would improve the average marks. Form a hypothesis for this statement.


# Null Hypothesis: The intervention of music to the class does not bring any change in the average marks of the class

# Alternate Hypothesis: The intervention of music to the class does bring change in the average marks of the class


# Problem: Class students have a mean score of 40 marks out of 100. The principal decided that some music while
# teaching would improve the performance of the class. Assume SD is 26.

# The class scored an average of 45 out of 100 after taking classes with the music. Can we be sure whether the
# increase in marks is a result of the music or is it just by random chance?


# Null Hypothesis: No significant difference in performance and the mean marks remains the same

# Alternate Hypothesis: Significant difference between the performance and the marks are different


# Z = New Mean - Old Mean / (SD / sqrt (N))

# Z = (45-40) / (26 / sqrt(100))

# Z = 5 / 2.6 = 1.92

# P value for Z Score 1.92 = 0.9726 (i.e. P value for marks greater than 40 = 1 - 0.9726 = 0.0274)
# P value for marks greater than 45 is 0.0274 which is significantly less than 0.05. Hence, we reject null hypothesis

# If the z test performed is a left tailed test "<", then this is the P-Value.  If, right tailed test ">" then P-Value
# is 1 minus this number.  If this is a two tailed test and the result is less than 0.5, then the double this number to
# get the P-Value.  If this is a two tailed test and the result is greater than 0.5 then first subtract from 1 and then
# double the result to get the P-Value.

# For a two tail test, Z score for 95% Probability is 1.96. For one tail test, 95% Probability is 1.645

# UNDERSTANDING ERRORS WHILE HYPOTHESIS TESTING

# TYPE 1 ERROR: When Null Hypothesis is actually True, but we rejected it. Also known as FALSE POSITIVE.
# TYPE 2 ERROR: When Null Hypothesis is actually False, but we failed to reject it. Also known as FALSE NEGATIVE.

# On other note, when we correctly reject a Null Hypothesis it is called Correct Rejection, and when we fail to reject
# the Null Hypothesis correctly, then it is called Correct Decision

""" UNDERSTANDING T TESTS """
# Link to t table: https://trainings.analyticsvidhya.com/courses/course-v1:AnalyticsVidhya+DS101+2018T2/courseware/9250e31ab1b84d13a0bec3cc00ecda14/bb1059b9e0f84c968cc749e9b1847512/?child=first

# So far we have been calculating the population SD, and calculating the critical value for the population distribution.
# However, if we want to find the test for the entire population from a sample, then a t-test is to be carried out.


# One tail test: If the critical value is to be checked on only one side of the mean (i.e. more or less than mean)
# For the above scenario for student, we need to see only if the marks had increased. So, for this particular case
# one tail test should be sufficient
# Two Tail Test: The critical value can lie on either side of the mean.

# The t distribution from the same will be wider than the normal distribution (from the population), since it represents
# only a portion of the population and is more prone to error

# CONDUCTING ONE SAMPLE T TEST

import pandas as pd
pd.set_option('display.max_column', 100)
pd.set_option('display.max_row', 110)
pd.set_option('expand_frame_repr', False)
project_dir = "C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course"
path = project_dir+"/Data Files/Understanding Statistics/One Tail T Test.csv"
import scipy.stats as stats
from scipy.stats import ttest_1samp

df = pd.read_csv(path)
df.head()

df.mean = df['Overall Marks'].mean()
t_statistic, p_value = ttest_1samp(df['Overall Marks'],70)
p_value
t_statistic

# As the P value is greater than 0.05, we fail to reject the null hypothesis (sample mean = population mean)
