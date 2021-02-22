import data_remover as dr
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, chi2_contingency, ttest_ind
import sklearn.metrics as sm
import seaborn as sns
import matplotlib.pyplot as plt

dr.data_remover()
# Load data set
data = pd.read_csv('dataset.csv')

# Name the columns
data.columns = ['complexAge', 'totalRooms', 'totalBedrooms',
                'complexInhabitants', 'apartmentsNr', 'medianComplexValue']

# Independent variables are stored in X
X = data[['complexAge', 'totalRooms', 'totalBedrooms',
          'complexInhabitants', 'apartmentsNr']].values
# # Dependent variable (price) is stored in Y
Y = data['medianComplexValue'].values  # values converts it into a numpy array

# Split the data into training/testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

corr = pd.DataFrame(data).corr()
plt.figure(figsize=(11, 11))
sns.heatmap(corr, cbar=1, square=1, annot_kws={'size': 15}, cmap= 'coolwarm')
plt.show()

regr = LinearRegression(normalize=True)  # creates object for the class
clf = regr.fit(X_train, Y_train)  # performs linear regression
Y_pred = regr.predict(X_test)  # makes predict
print('This is it', Y_pred)

# The coefficients
print('Slope coefficients: \n', regr.coef_)

print('Intercept: \n', regr.intercept_)
# The mean squared error
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f\n'
      % r2_score(Y_test, Y_pred))

print("Mean absolute error =", round(sm.mean_absolute_error(Y_test, Y_pred), 2))
print("Relative mean squared error =", round(sm.mean_squared_error(Y_test, Y_pred, squared=False), 2))
print("Median absolute error =", round(sm.median_absolute_error(Y_test, Y_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(Y_test, Y_pred), 2))
print("Max error =", round(sm.max_error(Y_test, Y_pred), 2))
print("R2 score =", round(sm.r2_score(Y_test, Y_pred), 2))

# Pearson’s Correlation Coefficient
# tests whether two samples have a linear relationship

data1 = Y_test
data2 = Y_pred
stat, p = pearsonr(data1, data2)
print('\nPearson’s Correlation Coefficient. stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably independent\n')
else:
    print('Probably dependent\n')

# Chi-Squared Test
# tests whether two categorical variables are related or independent

table = [data['apartmentsNr'], data['medianComplexValue']]
stat, p, dof, expected = chi2_contingency(table)
print('Chi-Squared Test. stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably independent\n')
else:
    print('Probably dependent\n')

# Student’s t-test
# tests whether the means of two independent samples are significantly different
data1 = Y_test
data2 = Y_pred
stat, p = ttest_ind(data1, data2)
print('Student’s t-test. stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably the same distribution\n')
else:
    print('Probably different distributions\n')

# Estimated multiple linear regression equation
# 𝑓(𝑥) = 𝑏₀ + 𝑏₁𝑥₁ + 𝑏₂𝑥₂ + 𝑏3𝑥3 + 𝑏4𝑥4 + b5x5

# x1 = input("Enter complexAge\n")
# x2 = input("Enter totalRooms\n")
# x3 = input("Enter totalBedrooms\n")
# x4 = input("Enter complexInhabitants\n")
# x5 = input("Enter apartmentsNr\n")
#
# inputs = [[x1, x2, x3, x4, x5]]
inputs = [[1, 7099, 1106, 2401, 1138]]  # test is 358500, predicted is 317416
prediction = regr.predict(inputs).flatten()

print("Predicted value", prediction)
