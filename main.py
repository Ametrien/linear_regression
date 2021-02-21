from sklearn.utils.extmath import safe_sparse_dot

import data_remover as dr
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, chi2_contingency, ttest_ind
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dr.data_remover()
# Load data set
data = pd.read_csv('dataset.csv')
# Name the columns
data.columns = ['complexAge', 'totalRooms', 'totalBedrooms',
                'complexInhabitants', 'apartmentsNr', 'medianComplexValue']

# Independent variables are stored in X
X = data[['complexAge', 'totalRooms', 'totalBedrooms',
          'complexInhabitants', 'apartmentsNr']].values
# Dependent variable (price) is stored in Y
Y = data['medianComplexValue'].values  # values converts it into a numpy array

# Split the data into training/testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

regr = LinearRegression()  # creates object for the class
regr.fit(X_train, Y_train)  # performs linear regression
Y_pred = regr.predict(X_test)  # makes predictions

# The coefficients
print('In multiple regression each coefficient is interpreted as the estimated change in y\n'
      'corresponding to a one unit change in a variable, when all other variables are held constant\n'
      'Slope coefficients: \n', regr.coef_)

print('Intercept: \n', regr.intercept_)
# The mean squared error
print('Mean absolute error: %.2f'
      % mean_absolute_error(Y_test, Y_pred))
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(Y_test, Y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f\n'
      % r2_score(Y_test, Y_pred))

# Estimated multiple linear regression equation
# 𝑓(𝑥) = 𝑏₀ + 𝑏₁𝑥₁ + 𝑏₂𝑥₂ + 𝑏3𝑥3 + 𝑏4𝑥4 + b5x5

x1 = input("Enter complexAge\n")
x2 = input("Enter totalRooms\n")
x3 = input("Enter totalBedrooms\n")
x4 = input("Enter complexInhabitants\n")
x5 = input("Enter apartmentsNr\n")

summ = 0
inputs = [x1, x2, x3, x4, x5]
for i in range (0, 5):
    for x in inputs:
       summ += regr.coef_[i] * int(x)

result = regr.intercept_ + summ
print("The medianComplexValue is", int(result))

# # Pearson’s Correlation Coefficient
# # tests whether two samples have a linear relationship
#
# data1 = Y_test
# data2 = Y_pred
# stat, p = pearsonr(data1, data2)
# print('Pearson’s Correlation Coefficient. stat=%.3f, p=%.3f' % (stat, p))
# if p > 0.05:
#     print('Probably independent\n')
# else:
#     print('Probably dependent\n')
#
# # Chi-Squared Test
# # tests whether two categorical variables are related or independent
#
# table = [data['apartmentsNr'], data['medianComplexValue']]
# stat, p, dof, expected = chi2_contingency(table)
# print('Chi-Squared Test. stat=%.3f, p=%.3f' % (stat, p))
# if p > 0.05:
#     print('Probably independent\n')
# else:
#     print('Probably dependent\n')
#
# # Student’s t-test
# # tests whether the means of two independent samples are significantly different
# data1 = Y_test
# data2 = Y_pred
# stat, p = ttest_ind(data1, data2)
# print('Student’s t-test. stat=%.3f, p=%.3f' % (stat, p))
# if p > 0.05:
#     print('Probably the same distribution\n')
# else:
#     print('Probably different distributions\n')
