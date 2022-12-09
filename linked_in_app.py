import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import plotly.graph_objects as go
from PIL import Image


st.title("WELCOME TO ABBY'S APP")
st.header("This app is used to predict if a user is a LinkedIn user based on varying predictors.")


income = st.selectbox("Gross Household Income level", 
             options = ["Less than $10,000",
                        "10,000 to under $20,000",
                        "20,000 to under $30,000",
                        "30,000 to under $40,000",
                        "40,000 to under $50,000",
                        "50,000 to under $75,000",
                        "75,000 to under $100,000",
                        "100 ,000 to under $150,000",
                        "$150,000 or more?"
                         ])
st.write(f"Income selected: {income}")

#st.write("**Convert Selection to Numeric Value**")

if income == "Less than $10,000":
   income = 1
elif income == "10,000 to under $20,000":
    income = 2
elif income == "20,000 to under $30,000":
     income = 3
elif income == "30,000 to under $40,000":
    income = 4
elif income == "40,000 to under $50,000":
    income = 5
elif income == "50,000 to under $75,000":
    income = 6
elif income == "75,000 to under $100,000":
    income = 7
elif income == "100,000 to under $150,000":
    income= 8
else:
    income = 9



education = st.selectbox("Highest Level Of Education Completed", 
             options = ["Less than high school",
                        "High School Incomplete",
                        "High school graduate",
                        "Some college, no degree",
                        "Two-year associate degree from a college or university",
                        "Four-year college or university degree/Bachelor's degree (e.g., BS, BA, AB)",
                        "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
                        "Postgraduate or professional degree, including master's, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)",
                         ])
st.write(f"Education selected: {education}")

#st.write("**Convert Selection to Numeric Value**")

if education == "Less than high school":
   education = 1
elif education == "High school incomplete":
    education = 2
elif education == "High school graduate":
     education = 3
elif education == "Some college, no degree":
    education = 4
elif education == "Two-year associate degree from a college or university":
    education = 5
elif education == "Four-year college or university degree/Bachelor's degree (e.g., BS, BA, AB)":
    education = 6
elif education == "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)":
    education = 7
else:
    education = 8



st.subheader("The foundations of the app are based on Machine Learning principles and Logistic Regression.")