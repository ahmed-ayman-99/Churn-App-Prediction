# Churn-App-Prediction
This project is part of my Data Mining training at ITI, where the goal of my work is to predict customer churn using a classification model.
By analyzing customer data, we aim to identify factors that contribute to churn and build a model capable of predicting which customers are likely to leave.

## Project Overview
This project utilizes a dataset containing the following features to predict customer churn:

* **Age:** The age of the customer.
* **Tenure:** The duration (in some unit, e.g., months or years) for which the customer has been a client.
* **Gender:** The gender of the customer.

The core steps involved in this project are:

1.  **Data Loading and Exploration:** Loading the dataset and performing initial exploratory data analysis to understand the data distribution and identify any potential issues.
2.  **Data Preprocessing:** Preparing the data for the classification model. This may involve handling missing values (if any) and encoding categorical features (like 'Gender') into a numerical format that the model can understand. Scaling numerical features ('Age', 'Tenure') might also be necessary.
3.  **Model Selection and Training:** Choosing an appropriate classification algorithm (e.g., Logistic Regression, Support Vector Machines, Random Forest, etc.) and training it on the prepared data.
4.  **Prediction:** Using the trained model to predict the likelihood of churn for new or unseen customer data.

## 🔧 Tools & Technologies Used:

* 🐍 **Python:** The primary programming language used for data analysis, preprocessing, and model building.
* 📈 **Matplotlib & Seaborn:** Python libraries used for creating insightful visualizations and exploring data patterns.
* 🧮 **pandas:** A powerful Python library for data manipulation and analysis, providing data structures like DataFrames.
* 📊 **Data preprocessing techniques:** Methods applied to clean and transform the raw data into a suitable format for the classification model. This includes techniques like handling missing values, encoding categorical variables, and scaling numerical features.
