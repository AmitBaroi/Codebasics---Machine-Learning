import pandas as pd
from sklearn import linear_model


"""
Multi-variate Linear Regression

Test cases:
case1: exp = 2, tst scr = 9, int scr = 8
case2: exp = 12, tst scr = 10, int scr = 10
"""

# Loading data
df = pd.read_csv(r"F:\_deep_\lessons\codebasics\data\2_C_hiring.csv")
# Renaming for easy access
df.rename({'experience': 'exp',
           'test_score(out of 10)': 'test',
           'interview_score(out of 10)': 'interview',
           'salary($)': 'salary'}, axis='columns', inplace=True)
# Replacing words for numbers
df.replace({'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11}, inplace=True)
# Filling in missing data
df.interpolate(inplace=True)
df.exp.fillna(0, inplace=True)

# Linear Regression
reg = linear_model.LinearRegression()
reg.fit(df[['exp', 'test', 'interview']], df.salary)

# Prediction
print(reg.predict([[2, 9, 8]]))
print(reg.predict([[12, 10, 10]]))
