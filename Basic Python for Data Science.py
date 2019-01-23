# PERFORMING A DO WHILE LOOP
i = 1
while i <= 5:
    print(i)
    i = i + 1

# CREATING A DICTIONARY IN PYTHON

room_num = {'john': 645, 'tom': 212}
print(room_num['tom'])  # print the value of the 'tom' key.
room_num['isaac'] = 345  # Add a new key 'isaac' with the associated value
print(room_num.keys())  # print out a list of keys in the dictionary
print('isaac' in room_num)  # test to see if 'issac' is in the dictionary.  This returns true.

# WRITING A IF STATEMENT

var = 5
if var > 0:
    print('Variable '+str(var)+' is positive')
elif var < 0:
    print('Variable is negative')
else:
    print('Variable is zero')


var = 8
if var % 2 == 0:
    print(str(var)+' is Even')
else:
    print(str(var)+' is Odd')

var = 100
if var > 90:
    print('Grade A')
elif var > 60:
    print('Grade B')
else:
    print('Grade F')

range(5)

# USING FOR LOOP

for i in range(5):
    print(i)

for i in '':
    print(i)

# PRINT ALL NUMBERS BETWEEN 10 AND 20

for i in range(10, 20):
    print(i)

# PRINT ALL ODD NUMBERS BETWEEN 10 AND 20

for i in range(10, 21):
    if i % 2 != 0:
        print(i)

# ALTERNATE WAY TO CALCULATE THE ODD NUMBERS BETWEEN 10 AND 20 USING RANGE INCREMENT OF 2

for i in range(11, 21, 2):
    print(i)

# CREATE A FUNCTION TO FIND THE GREATER OF TWO NUMBERS


def greater_number(a, b):
    if a > b:
        print(a)
    else:
        print(b)


greater_number(1, 10)


# ALTERNATE WAY TO WRITE THE ABOVE FUNCTION TO RETURN A VALUE


def greater_number(a, b):
    if a > b:
        greater = a
    else:
        greater = b
    return greater


greater_number(10, 20)

# WORKING WITH LISTS IN PYTHON

family_list = ['Sivai', 'Amrita', 'Paplu', 'Divya']

# RETRIEVE THE SECOND OBJECT FROM THE LIST
# NOTE THAT THE LIST INDEX ALWAYS STARTS WITH 0
family_list[1]

# RETRIEVE FIRST THREE THE OBJECTS IN THE LIST

family_list[0:2]

# ADDING SINGLE OBJECT TO THE LIST

family_list.append('Shanthi')  # family_list

# ADDING MULTIPLE OBJECTS TO A LIST

family_list.extend(['Chandrasekar', 'Kuppubabu', 'Sujatha', 'Sarengapani'])  # family_list | family_list[8]

# REMOVING AN OBJECT FROM A LIST
# REMOVING OBJECT USING THE ACTUAL VALUE

family_list.remove('Sivai')  # family_list
family_list
del family_list[0]
family_list

# LISTS CAN BE USED IN LOOPING

for i in family_list:
    print(i)

""" CREATING A DICTIONARY """

marks = {'history': 76, 'Geography': 77, 'Mathematics': 88}
marks

marks['Geography']  # ACCESS ELEMENTS IN A DICTIONARY
marks['English'] = 89  # ADDING ELEMENTS
marks.update({'Physics': 100, 'Chemistry': 89})  # ADDING MULTIPLE ITEMS TO DICTIONARY
del marks['Geography']  # Deleting objects from a dictionary
marks

# FOR REMOVING MULTIPLE DICTIONARIES

to_remove = ['history', 'Mathematics']
for i in to_remove:
    marks.pop(i, None)

marks



""" UNDERSTANDING THE CONCEPT OF STANDARD LIBRARIES """

# Standard libraries are functions that are available in Python Base Package
# Modules are packages installed in to python to perform extended functions that base python
# cannot handle or not good at handling

# from Operator.Arithmetic import addition
# Here Operator is the package, and Arithmetic is module, and addition is function

""" READING CSV FILES IN PYTHON"""

# Pandas is a package for performing data analysis tasks in python. It can perform reading,
# filtering, manipulating, visualizing, and exporting data
# Pandas can import csv, excel, json, html, local clipboard, stata, sql  etc.

import pandas as pd   # Importing Pandas library
project_dir = "C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course"
path = project_dir+"/Data Files/Basic Python for Data Science/Reading CSV File in Python.csv"
df = pd.read_csv(path)
df.head()

path1 = project_dir+"/Data Files/Basic Python for Data Science/Reading CSV File in Python.xlsx"
df1 = pd.read_excel(path1)
df1.head()

"""UNDERSTANDING DATA FRAMES AND BASIC OPERATIONS"""

df.shape  # To get the number of rows and columns
df.head(5)  # Displays the top n rows in a data frame
df.tail(3)  # Displays the bottom n rows in a data frame
df.columns  # Displays all the column names
df["Country Name"]  # Accessing a column in a data frame
df[["Country Name", "1980"]]  # Accessing multiple columns in a data frame

import pandas as pd   # Importing Pandas library
pd.set_option('display.max_columns', 100)
pd.set_option('expand_frame_repr', False)
path = project_dir+"/Data Files/Basic Python for Data Science/SampleData.csv"
df = pd.read_csv(path)
df.shape
df.head()
df.tail()
df.columns
df['writing score']
df[['writing score', 'math score']]
df.iloc[:7, :4]  # iloc function is used to refer index data frame using row and col position
df[df['lunch'] == "standard"]  # Subset rows using a specific value

# EXERCISE PROBLEM

df = pd.read_csv(project_dir+"/Data Files/Basic Python for Data Science/index.csv")
df.head()
df['8']  # Accessing the third column in the data frame
df.iloc[:, 2]  # Alternate way to access the third column in the data frame
df.columns
df.iloc[:, -2:]  # Access the last two columns in the data frame
df.iloc[-10:, :2]  # Access the last 10 rows and first two columns

# PYTHON SKILLS EVALUATION PROBLEM

df = pd.read_csv(project_dir+"/Data Files/Basic Python for Data Science/data_python.csv")
df.head()
print(df)
df.iloc[24, 4]  # Access the 25th row and the 5th column
df.iloc[24:25, 4:5]  # Alternate way to access the 25th row and the 5th column
df.iloc[:, 3:5]
df.loc[:, ['Dependents', 'Education']]
df.ix[:, ['Dependents', 'Education']]

"""PYTHON_CODING_CHALLENGE"""

# Demo
# initialize variable 'msg' with the string 'Hello World'
msg = "Hello World"

# initialize variables 'a' and 'b' with 5 and 6 respectively
a = 5
b = 6

# add 'a' and 'b' and assign the result into a new variable 'c'
c = a + b
print(c)

# build a function to add 2 numbers


def addition(x, y):
    return(x + y)

# use the function 'addition' to add 'a' and 'b'


addition(a, b)

# create a list consisting of first 5 even numbers and print the list
my_list = [2, 4, 6, 8, 10]
print(my_list)

# access the 3rd element of the list 'my_list'
my_list[2]

# given below is a dictionary having 4 unique keys, i.e., 'name', 'age', 'gender', 'is_employed'
my_dict = {'name': 'Smith',
           'age': 34,
           'gender': 'Male',
           'is_employed': False}

# print 'my_dict'
print(my_dict)

# access value under 'name' key from 'my_dict'
my_dict['name']

# update 'is_employed' key to True
my_dict.update({'is_employed': True})

# print the updated dictionary
print(my_dict)

# use a for loop to print only even numbers from the first 20 numbers, i.e. 1-20
for i in range(1,21):
    if i % 2 == 0:
        print(i)

for i in range(2, 21, 2):
    print(i)

# import required libraries
import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('expand_frame_repr', False)
# import numpy as np

# read data_python.csv using pandas
mydata = pd.read_csv(project_dir+"/Data Files/Basic Python for Data Science/data_python.csv")

# print the number of rows and number of columns of mydata
mydata.shape
mydata.head()
# assign a variable 'target' with the 'Loan_Status' feature from mydata dataframe
target = mydata['Loan_Status']

# print the datatype of ApplicantIncome feature
print(mydata['ApplicantIncome'].dtype)

# conditional statement - print 'Yes' if the 21st element of 'Education' feature is 'Graduate' else print 'No'
if mydata['Education'][20] == 'Graduate':
    print('Yes')
else:
    print('No')


# print 31st to 35th rows of mydata
mydata.iloc[31:36]


# print first 5 rows of 2nd and 3rd column only
mydata.iloc[:5, 1:3]


