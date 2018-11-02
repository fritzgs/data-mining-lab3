"""

Method of showing Regression model taken from Alice ChiaHui Lui (2018)
https://www.kaggle.com/chiahuiliu/avocado-exploratory-and-regression/notebook

FRITZ GERALD SANTOS - 20071968
Building a regression model

"""

import pandas
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import explained_variance_score

#Import the dataset
dataframe = pandas.read_csv("avocado.csv")

#Cleaning the data
dataframe.rename(columns={'year': 'Year', 'region' : 'Region', 'type' : 'Type'}, inplace=True)

#cleaning - convert the data type of date into datetime.
dataframe["Date"] = pandas.to_datetime(dataframe["Date"])

#Cleaning - convert the data type of region to category
dataframe['Region'] = dataframe['Region'].astype('category')
dataframe['Region'] = dataframe['Region'].cat.codes

#Create dummy variables for type series - have two separate series for conventional and organic (representing 1 if true and 0 if false)
dummy_type = pandas.get_dummies(dataframe['Type'])
dataframe = pandas.concat([dataframe, dummy_type], axis=1)

#Split the dates into quarters of the year.
dataframe['Date_Q'] = dataframe['Date'].apply(lambda x: x.quarter)

#Create X and Y columns
X_columns = ['Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 'Year','Region', 'conventional', 'organic', 'Date_Q']
X = dataframe[X_columns]
Y = dataframe['AveragePrice']

#Create X and Y train to use for model
X_train, X_test, y_train, y_test  = train_test_split(X, Y, test_size=0.33, random_state=2018)

#For the regression model.
model = sm.OLS(y_train, X_train)
res = model.fit()
print(res.summary()) #print the regression model.
