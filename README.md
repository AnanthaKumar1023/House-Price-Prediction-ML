# House Price Prediction System

## Project Overview

This project uses Machine Learning to predict house prices based on property-related features such as area, bedrooms, bathrooms, age, and location.

The project uses Linear Regression to train a model and predict the estimated price of a new house.

## Features Used

* Area
* Bedrooms
* Bathrooms
* Age
* Location

## Machine Learning Algorithm

### 1. Linear Regression

Used to predict the price of houses based on the given property features.

## Data Preprocessing

* Missing value handling
* Categorical data encoding using One-Hot Encoding
* Train-Test Split
* Feature and target separation

## Dataset

The dataset contains house property records with the following information:

* Area
* Bedrooms
* Bathrooms
* Age
* Location
* Price

## Model Evaluation

The model is evaluated using:

* MAE (Mean Absolute Error) :  ₹285,396.29
* MSE (Mean Squared Error) : 177,990,372,716.59
* RMSE (Root Mean Squared Error) : ₹421,889.05
* R² Score :  0.9812

## Project Outputs

* Actual vs Predicted House Prices
* Actual vs Predicted Scatter Plot
* New House Price Prediction

## Example Prediction

The model predicts the estimated price for a new house using:

* Area: 1800 sq.ft
* Bedrooms: 3
* Bathrooms: 2
* Age: 5 years
* Location: Chennai

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## Future Scope

* Use a larger real-world dataset
* Try advanced Machine Learning algorithms
* Compare different regression models
* Improve prediction accuracy
* Develop a web-based house price prediction system
* Deploy the model for real-time predictions
