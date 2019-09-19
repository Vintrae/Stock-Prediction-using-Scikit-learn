# Stock Prediction using Scikit-learn

In this project the aim is to model stock price for prediction of price based on past data. You can see the program running in the following [Colaboratory notebook](https://colab.research.google.com/drive/125se_adfV5pdgygfHzWy8Np8HTMExaDN).

To do this task I used 3 different types of regression models to see which one provides more accuracy:

- Least-squares linear regression. This simple approach to linear regression uses the **sum of least squares** approach to analyse the data to determine the output variable, which is stock price in this case.


- Ridge regression. This type of linear regression is optimal to analyse data that suffers from multicollinearity (i.e. variables are not independent but highly related instead). In order to do so it **adds a degree of bias** to the regression estimates, which reduces the standard error that least squares has for these cases.


- Lasso regression: this type of regression uses **shrinkage**, where data values are made smaller towards a centrail point (e.g. the mean). This is ideal for models with fewer parameters (sparse).

In order to do this we are going to use the **Scikit-learn** library together with **Pandas** for data-structure representation and **Matplotlib** to plot the resulting data. I have divided the process into the following steps.

* STEP 1. Retrieving stock data and formatting.
* STEP 2. Preprocessing the data and separating independent variables from labels.
* STEP 3. Creating regression models and checking performance.
* STEP 4. Visualise predictions in graphs using matplotlib.
