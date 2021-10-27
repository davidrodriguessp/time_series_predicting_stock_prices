'''
PREDICTING NEXT DAY STOCK MARKET PRICES
Author: David Rodrigues - davidrodriguessp@hotmail.com | https://www.linkedin.com/in/davidrodrigues/

This projet aimed to create a model to predict stock prices based on historic data.
We used data available in the file sphist.csv with S&P500 prices from1950 to 2015.

The dataset has 16590 rows and 7 columns.

The data can be found online in Kaggle in the link:
https://www.kaggle.com/samaxtech/sp500-index-data?select=sphist.csv

The time series nature of the data means that we can generate indicators to make our model
more accurate. For instance, you can create a new column that contains the average price of the
last 10 trades for each row. This incorporates information from multiple prior rows into one and
makes predictions more accurate.
'''

# Import pandas and read the dataset into a Pandas DataFrame
import pandas as pd
pred = pd.read_csv("sphist.csv")

# Covert the Date column into datetime
pred['Date'] = pd.to_datetime(pred['Date'])

# Count how many rows we have before 2013
from datetime import datetime
before_2013 = pred['Date'] < datetime(year=2013, month=1, day=1)

# Sort data from older to newer
pred = pred.sort_values('Date', ascending=True)

# Create a new column with avg of past 5 days
pred['avg_past5'] = (
    pred['Close'].rolling(5).mean().shift()
                                    )
# Create a new column with avg of past 365 days
pred['avg_past365'] = (
    pred['Close'].rolling(365).mean().shift()
                                        )
# Create a new column with std of past 5 days
pred['std_past5'] = (
    pred['Close'].rolling(5).std().shift()
                                    )

# Create a new column with ratio between the
# avg price past 5 days and avg price past 365 days
pred['ratio_past5_past365'] = (
    pred['avg_past5'] / pred['avg_past365']
                                                        )
# Remove all rows that occur before 1951-01-03
from datetime import datetime
pred = pred[pred['Date'] > datetime(year=1951,
                                    month=1, day=2)]

# Remove all remaining missing values
pred = pred.dropna()

'''
SPLIT THE DATASET INTO TRAIN AND TEST
- Train set will include all data before 2013.
- Test set will include all data from 2013 onwards.
'''

# Create the train DataFrame
train = pred[pred['Date'] < datetime(year=2013,
                                     month=1, day=1)]

# Create the test DataFrame
test = pred[pred['Date'] >= datetime(year=2013,
                                     month=1, day=1)]

# We will consider MAE - Mean Absolute Error - as the main error metric
# Initialize an instance of the linear regression class
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# Train a model using the train data.
# We have NOT used current data as predictors as they contain knowledge of the future
# We used Close as the target column
# Use the new created indicators as predictors

# Spit data into X_train, X_test, y_train, y_test
X_train = train[['avg_past5', 'avg_past365',
                 'std_past5', 'ratio_past5_past365']]
X_test = test[['avg_past5', 'avg_past365',
                 'std_past5', 'ratio_past5_past365']]
y_train = train['Close']
y_test = test['Close']

# Fit the model with the train dataset
lr.fit(X=X_train, y=y_train)

# Make predictions for the target column in test set
predictions = lr.predict(X=X_test)

# Calculate the Mean Absolute Error
from sklearn.metrics import mean_absolute_error
y_true = test['Close'] # create set of true values to calculate MAE
y_pred = predictions # create a set of predicted values to calculate MAE

# Calculate MAE
mae = mean_absolute_error(y_true, y_pred)

'''
The MAE calculated is 16.09
Let's see how this compares with the mean price in the test set.
'''

# Calculating the mean price of the test set
mean_price = test['Close'].mean()

print(
    ' Mean Absolute Error: ', mae, '\n',
    'Mean Price: ', mean_price, '\n',
    'Ration MAE / Mean Price: ',
    round(mae / mean_price,4)
    )

# Calculate the R2 of the model
r2 = lr.score(X_test, y_test)
print(' R2: ', round(r2, 3))

'''As observed as we run the script, we find
 Mean Absolute Error:  16.093177677832042 
 Mean Price:  1874.8903383897166 
 Ration MAE / Mean Price:  0.0086
 R2:  0.99%

 The model seems very accurate in predicting the next day price.
 If we consider the true and predicted prices, prices are predicted with an accuracy of 99%.
 '''
