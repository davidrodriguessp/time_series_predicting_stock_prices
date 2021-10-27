# PREDICTING NEXT DAY STOCK MARKET PRICES
*Author: David Rodrigues - davidrodriguessp@hotmail.com | https://www.linkedin.com/in/davidrodrigues/*

This projet aimed to create a model to predict stock prices based on historic data.
We used data available in the file sphist.csv with S&P500 prices from1950 to 2015.

The dataset has 16590 rows and 7 columns.

The data can be found online in Kaggle in the link:
https://www.kaggle.com/samaxtech/sp500-index-data?select=sphist.csv

The time series nature of the data means that we can generate indicators to make our model
more accurate. For instance, you can create a new column that contains the average price of the
last 10 trades for each row. This incorporates information from multiple prior rows into one and
makes predictions more accurate.

This repository has two files:
- predict.py with the full code
- sphist.csv with the dataset 

