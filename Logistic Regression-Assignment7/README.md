# Titanic Survival Prediction using Logistic Regression

## Project Overview
This project predicts whether a passenger survived the Titanic disaster using a Logistic Regression model.  
The goal is to build a classification model, interpret it, evaluate it properly, and deploy it using Streamlit.

## Dataset
- Titanic_train.csv – training dataset  
- Titanic_test.csv – test dataset  

Target variable: **Survived**

## Approach

1. Data Understanding & EDA  
   - Explored survival distribution, age distribution, class vs survival, and gender vs survival  
   - Identified class imbalance and key survival drivers  

2. Data Preprocessing  
   - Dropped irrelevant features (PassengerId, Name, Ticket, Cabin)  
   - Handled missing values using median imputation by passenger class  
   - One-hot encoded categorical variables  
   - Scaled numerical features using StandardScaler  

3. Modeling  
   - Trained a Logistic Regression classifier  
   - Tuned the probability threshold for recall vs precision tradeoff  
   - Evaluated using Precision, Recall, F1-score, ROC Curve  

4. Interpretation  
   - Interpreted coefficients to understand survival drivers  
   - Identified key predictors: Sex, Pclass, Age, Fare  

5. Deployment  
   - Built a Streamlit app for interactive prediction  
   - Allows users to view test data and predict survival probability  

## Key Findings

- Females had significantly higher survival probability than males  
- First class passengers survived more than lower classes  
- Younger passengers had better survival chances  
- Threshold tuning improved recall for survivor detection  

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
