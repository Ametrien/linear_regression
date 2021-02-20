import data_remover as dr
from sklearn.model_selection import train_test_split
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, chi2_contingency, ttest_ind

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
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean absolute error: %.2f'
      % mean_absolute_error(Y_test, Y_pred))
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(Y_test, Y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f\n'
      % r2_score(Y_test, Y_pred))

# Pearson’s Correlation Coefficient
# tests whether two samples have a linear relationship

data1 = Y_test
data2 = Y_pred
stat, p = pearsonr(data1, data2)
print('Pearson’s Correlation Coefficient. stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably independent\n')
else:
    print('Probably dependent\n')

# Chi-Squared Test
# tests whether two categorical variables are related or independent

table = [data['totalRooms'], data['medianComplexValue']]
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
