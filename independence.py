"""
Lab 3 - independence test
FRITZ GERALD SANTOS - 20071968

"""
import pandas
import numpy as np
from scipy import stats


dataframe = pandas.read_csv("avocado.csv")
dataframe.rename(columns={'year': 'Year', 'region' : 'Region', 'type' : 'Type'}, inplace=True)

#Frequency table showing how many times the price equaled x in that region.
contingency_table = pandas.crosstab(dataframe.AveragePrice, dataframe.Region, margins=True)

#Observed data saved into an array.
f_obs = np.array([contingency_table.values])


print(stats.chi2_contingency(f_obs)[0:3])
