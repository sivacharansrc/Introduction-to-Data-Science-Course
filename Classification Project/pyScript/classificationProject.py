import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('expand_frame_repr', False)
import matplotlib.pyplot as plt

### Reading the data set

df = pd.read_csv("C:/Users/sivac/Documents/Python Projects/Introduction to Data Science Course/Classification Project/input/Train.csv")
df.head()