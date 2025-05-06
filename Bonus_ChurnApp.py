##Ahmed Ayman Hassan Sobieh

import pandas as pd
import numpy as np
import joblib as jb
import streamlit as st


st.title("Churn Prediction App")
st.subheader("Created By: Ahmed Ayman Hassan 😎")
st.write("Hello, This is a simple app to predict customer churn using a machine learning model")


Churn_App = pd.read_excel("churn_dataset.xlsx")


st.subheader("📊 Dataset Preview")
st.write(Churn_App.head())


model = jb.load ("NaiveBayes.pkl")


st.sidebar.header("Input customer data")
age = st.sidebar.slider("Customer Age",int(Churn_App["Age"].min()),int(Churn_App["Age"].max()),int(Churn_App["Age"].mean()))
Tenure = st.sidebar.slider("Tenure",int(Churn_App["Tenure"].min()),int(Churn_App["Tenure"].max()),int(Churn_App["Tenure"].mean()))
Gender = st.sidebar.radio('Gender',['Male','Female'] )
Gender_ = 0 if Gender == 'Male' else 1


input_data = np.array([[age, Tenure,Gender_]])

Prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)[0]

st.subheader("Prediction Results")
special_labels ={0:"Not Churn", 1:"Churn"}
st.write("Prediction: ", special_labels[Prediction[0]])

st.write(f'Probability: {prediction_proba[0]:.2%}  {special_labels[Prediction[0]]}')