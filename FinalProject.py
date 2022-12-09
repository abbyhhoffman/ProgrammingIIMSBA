#!/usr/bin/env python
# coding: utf-8

# # Final Project
# ## Abby Hoffman
# ### December 13, 2022

# ***

# In[265]:


import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# #### Q1

# Read in the data, call the dataframe "s"  and check the dimensions of the dataframe

# In[8]:


s = pd.read_csv("social_media_usage.csv")


# In[12]:


s.shape


# There are 1502 Rows and 89 columns.

# ***

# #### Q2

# Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected

# In[176]:


toy_df = pd.DataFrame({'A':[1,2,3],
                       'B':[4,5,6]})


# In[177]:


toy_df


# In[185]:


def clean_sm(data):
    x = (np.where(data==1,1,0))
    return x


# In[186]:


clean_sm(toy_df)


# ***

# #### Q3

# Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.

# In[208]:


ss = pd.DataFrame({
    "sm_li":clean_sm(s["web1h"]),
    "income":np.where(s["income"] > 9, np.nan, s["income"]),
    "education":np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent":np.where(s["par"] == 1, 1, 0),
    "married":np.where(s["marital"] == 1, 1, 0),
    "female":np.where(s["gender"] >= 3, np.nan,
                          np.where(s["gender"] == 2, 1, 0)),
    "age":np.where(s["age"] > 98, np.nan, s["age"])}).dropna().sort_values(by=["income","education"], ascending=True)


# In[209]:


ss # education and income are in ascending order. 
    # all variables are accounted for , no missing values 


# #### Exploratory Data Analysis

# In[218]:


alt.Chart(ss.groupby(["age", "education"], as_index=False)["sm_li"].mean()).mark_circle().encode(x="age",
      y="sm_li",
      color="education:N").interactive()


# In[220]:


alt.Chart(ss.groupby(["income", "parent"], as_index=False)["sm_li"].mean()).mark_circle().encode(x="income",
      y="sm_li",
      color="parent:N")


# In[222]:


alt.Chart(ss.groupby(["income", "age"], as_index=False)["sm_li"].mean()).mark_circle().encode(x="age",
      y="sm_li",
      color="income:N")


# In[224]:


alt.Chart(ss.groupby(["income", "education"], as_index=False)["sm_li"].mean()).mark_circle().encode(x="income",
      y="sm_li",
      color="education:N")


# ***

# #### Q4

# Create a target vector (y) and feature set (X)

# In[226]:


# Target (y) and feature(s) selection (X)
y = ss["sm_li"]
X = ss[["age", "education", "female", "income", "married", "parent"]]


# ***

# #### Q5

# Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning

# In[227]:


# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=987) # set for reproducibility

# X_train contains 80% of the data and contains the features used to predict the target when training the model. 
# X_test contains 20% of the data and contains the features used to test the model on unseen data to evaluate performance. 
# y_train contains 80% of the the data and contains the target that we will predict using the features when training the model. 
# y_test contains 20% of the data and contains the target we will predict when testing the model on unseen data to evaluate performance.


# ***

# #### Q6 

# Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.

# In[229]:


# Initialize algorithm 
lr = LogisticRegression(class_weight= "balanced")


# In[230]:


# Fit algorithm to training data
lr.fit(X_train, y_train)


# ***

# #### Q7

# Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.

# In[231]:


# Make predictions using the model and the testing data
y_pred = lr.predict(X_test)


# In[238]:


confusion_matrix(y_test, y_pred)


# #MISSING ANSWER

# ***

# #### Q8

# Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents

# In[243]:


# Compare those predictions to the actual test data using a confusion matrix (positive class=1)

pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap='RdYlBu')


# #MISSING ANSWER

# ***

# #### Q9

# Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. Use the results in the confusion matrix to calculate each of these metrics by hand. Discuss each metric and give an actual example of when it might be the preferred metric of evaluation. 

# In[250]:


## recall: TP/(TP+FN)
65/(65+19)


# #MISSING ANSWER: Discuss each metric and give an actual example of when it might be the preferred metric of evaluation.

# In[251]:


## precision: TP/(TP+FP)
65/(65+56)


# #MISSING ANSWER: Discuss each metric and give an actual example of when it might be the preferred metric of evaluation.

# In[252]:


## F1: Weighted average of recall and precision (2*((precision*recall)/precisions +recall)

2*((0.7738095238095238*0.5371900826446281)/(0.5371900826446281+0.7738095238095238))


# #MISSING ANSWER: Discuss each metric and give an actual example of when it might be the preferred metric of evaluation.

# After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.

# In[237]:


print(classification_report(y_test, y_pred)) #hand calculated match! 


# ***

# #### Q10

# Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn?

# In[256]:


# New data for features: age, education, female, high_income, married, parent 
person = [42, 7, 1, 8, 1, 0]

# Predict class, given input features
predicted_class = lr.predict([person])

# Generate probability of positive class (=1)
probs = lr.predict_proba([person])


# In[263]:


# Print predicted class and probability
print(f"Predicted class: {predicted_class[0]}") # 0=not a user, 1= user
print(f"Probability that this person is a LinkedIn User: {probs[0][1]}")


# How does the probability change if another person is 82 years old, but otherwise the same?

# In[261]:


# New data for features: age, education, female, high_income, married, parent 
person2 = [82, 7, 1, 8, 1, 0]

# Predict class, given input features
predicted_class2 = lr.predict([person2])

# Generate probability of positive class (=1)
probs2 = lr.predict_proba([person2])


# In[262]:


# Print predicted class and probability
print(f"Predicted class: {predicted_class2[0]}") # 0=not a user, 1= user
print(f"Probability that this person is a LinkedIn User: {probs2[0][1]}")


# DESCRIPTION OF DIFFERENCE IN PROBABILITY MUST ANSWER
