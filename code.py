import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
# from ISLP import datasets
from ISLP.models import (ModelSpec as MS, summarize, poly)
# Load the 'insurance' dataset from ISLP
df = pd.read_csv('./insurance.csv')

# 1. Exploratory Data Analysis (EDA)

# Check the first few rows of the dataset
print("Dataset Overview:")
print(df.head())

# Check summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Visualize the distribution of charges
plt.figure(figsize=(8, 6))
sns.histplot(df['charges'], kde=True)
plt.title('Distribution of Charges')
plt.show()

# Visualize the relationships between charges and other variables
sns.pairplot(df)
plt.suptitle('Pairplot of Variables', y=1.02)
plt.show()

print('- - - - - - - - - - - - - - - - - - - - - - - -\n 1. Is there a relationship between charges and the attributes?\n2. How strong is the relationship? \nLinear Regression Analysis:')

# Choose the dependent variable (response variable)
y = df['charges']

# Choose independent variables
X = MS(['age', 'sex', 'smoker', 'region', 'bmi', 'children']).fit_transform(df)

# Add categorical variables as dummy variables
X = pd.get_dummies(X, columns=['sex', 'smoker', 'region'], drop_first=True)

# Add a constant term to the independent variables (for the intercept)
X = sm.add_constant(X)

# Fit a linear regression model
model = sm.OLS(y, X.astype(float)).fit()
# Display the summary statistics
print("\nLinear Regression Results:\n", model.summary())
print('- - - - - - - - - - - - - - - - - - - - -\nSummarize Model:\n')
summarize(model)
# Answering the questions
print('- - - - - - - - - - - - - - - - - - - - - - - -\n3. Which attributes are associated with charges?')
print('4. How strong is the association between each attribute and charges?')
print("\nAssociation with Charges:")
print(model.params)

print('- - - - - - - - - - - - - - - - - - - - - - - -\n5. How accurately can we predict future charges?')
# Display the R-squared value
print(f"\nR-squared: {model.rsquared}")

print('- - - - - - - - - - - - - - - - - - - - - - - -\n6. Is the relationship linear? yes. check the plot!')
plt.figure(figsize=(8,8))
plt.scatter(model.fittedvalues, model.resid)
plt.xlabel('Fitted value')
plt.ylabel('Residual')
plt.axhline(0, c='k', ls='--')
plt.show()

print('- - - - - - - - - - - - - - - - - - - - - - - -\n7. Is there synergy among the attributes?')
# Display VIF (Variance Inflation Factor) to check for multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF

vals = [VIF(X.astype(float), i)
        for i in range(1, X.shape[1])]
vif = pd.DataFrame({'vif':vals},
                   index=X.columns[1:])
print("\nVIF:")
print(vif)
