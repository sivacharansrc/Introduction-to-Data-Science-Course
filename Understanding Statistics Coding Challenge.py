# import required libraries
import pandas as pd
import numpy as np

scores = [29, 27, 14, 23, 29, 10]

# find the mean of all items of the list 'scores'
np.mean(scores)

# find the median of all items of the list 'scores'
np.median(scores)

# Find the mode of the list fruits
from statistics import mode
fruits = ['apple', 'grapes', 'orange', 'apple']

# find mode of the list 'fruits'
mode(fruits)



from random import sample
data = sample(range(1, 100), 50)    # generating a list 50 random integers

# find variance of data
np.var(data)


# find standard deviation
np.std(data)
(np.var(data))**0.5

# read data_python.csv using pandas
project_dir = "C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course"
path = project_dir+"/Data Files/Understanding Statistics/data_statistics.csv"
mydata = pd.read_csv(path)

# print first few rows of mydata
mydata.head()

# plot histogram for 'Item_Outlet_Sales'
plt.hist(mydata['Item_Outlet_Sales'])
plt.show()

# increadse no. of bins to 20
plt.hist(mydata['Item_Outlet_Sales'], bins=20)
plt.show()

# find mean and median of 'Item_Weight'
np.mean(mydata['Item_MRP']), np.median(mydata['Item_MRP'])

# find mode of 'Outlet_Size'
mydata['Outlet_Size'].mo
mode(mydata['Outlet_Size'])

# frequency table of 'Outlet_Type'
mydata['Outlet_Type'].value_counts()

# mean of 'Item_Outlet_Sales' for 'Supermarket Type2' outlet type
np.mean(mydata['Item_Outlet_Sales'][mydata['Outlet_Type'] == 'Supermarket Type2'])


# mean of 'Item_Outlet_Sales' for 'Supermarket Type3' outlet type
np.mean(mydata['Item_Outlet_Sales'][mydata['Outlet_Type'] == 'Supermarket Type3'])

# 2 sample independent t-test
from scipy import stats
stats.ttest_ind(mydata['Item_Outlet_Sales'][mydata['Outlet_Type'] == 'Supermarket Type2'], mydata['Item_Outlet_Sales'][mydata['Outlet_Type'] == 'Supermarket Type3'])
