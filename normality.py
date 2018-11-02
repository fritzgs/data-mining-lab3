"""
FRITZ GERALD SANTOS - 20071968
Testing for Normality

"""

import pandas
import seaborn
import matplotlib.pyplot as pyplot
from scipy import stats

#Import the dataset
dataframe = pandas.read_csv("avocado.csv")
dataframe.rename(columns={'year': 'Year', 'region' : 'Region', 'type' : 'Type'}, inplace=True)


#Show the distribution of the average price data - normal data would result in bell shape visual.
seaborn.distplot(dataframe.AveragePrice, kde=True, rug=True)
pyplot.show()

#To support the above, if the result of the probability plot is a linear shape, the data is normal.
stats.probplot(dataframe.AveragePrice, dist="norm", plot=pyplot)
pyplot.show()