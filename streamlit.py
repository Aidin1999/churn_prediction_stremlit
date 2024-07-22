import streamlit as st 
import pandas as pd 
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
import joblib  # To load the pre-trained model

# Title of the app
st.title('Bank Customer Churn Prediction')

st.write("This web app predicts whether a customer will leave the bank.")

# Sidebar for user inputs
st.sidebar.header('User Inputs')

def features():
    # Get user inputs from sidebar
    Creditscore = st.sidebar.slider('Credit Score', 350, 850, 650)
    Tenure = st.sidebar.slider('Tenure (Years)', 0, 10, 5)
    Balance = st.sidebar.slider('Balance ($)', 0, 200000, 50000)
    NumOfProducts = st.sidebar.slider('Number of Products', 1, 4, 2)
    EstimatedSalary = st.sidebar.slider('Estimated Salary ($)', 0, 200000, 100000)
    Age = st.sidebar.slider('Age', 18, 92, 38)
    HasCrCard = st.sidebar.selectbox('Has Credit Card', ['Yes', 'No'])
    IsActiveMember = st.sidebar.selectbox('Is Active Member', ['Yes', 'No'])
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    Geography = st.sidebar.radio('Geography', ['France', 'Spain', 'Germany'])
    
    # Convert categorical inputs to numerical
    HasCrCard = 1 if HasCrCard == 'Yes' else 0
    IsActiveMember = 1 if IsActiveMember == 'Yes' else 0
    gender = 1 if gender == 'Male' else 0
    Geography = {'France': 1, 'Spain': 2, 'Germany': 3}.get(Geography, 1)
    
    data = {'CreditScore': Creditscore,
            'Geography': Geography,
            'Gender': gender,
            'Age': Age,
            'Tenure': Tenure,
            'Balance': Balance,
            'NumOfProducts': NumOfProducts,
            'HasCrCard': HasCrCard,
            'IsActiveMember': IsActiveMember,
            'EstimatedSalary': EstimatedSalary}
    
    features = pd.DataFrame(data, index=[0])
    return features

# Load pre-trained model
model_path = 'xgb_model.pkl' 
xgb_model = joblib.load(model_path) 

scaler_path = 'ss.pkl'
ss = joblib.load(scaler_path)


# User input parameters
df = features()
st.subheader('User Input Parameters')
st.write(df)

# Scale user input and make prediction
df_scaled = ss.transform(df)
prediction = xgb_model.predict(df_scaled)

# Display prediction result
if prediction[0] == 1:
    st.write('The customer is likely to leave the bank.')
else:
    st.write('The customer is not likely to leave the bank.')
