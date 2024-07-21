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
model_path = 'xgb_model.pkl'  # Change this to the correct path
xgb_model = joblib.load(model_path)

# Load and prepare data
data = pd.read_csv('Churn.csv')
data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
data['Geography'] = data['Geography'].map({'France': 1, 'Spain': 2, 'Germany': 3})

x = data.iloc[:, :-1]
y = data['Exited']

# Split and scale data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=85)
ss = StandardScaler()
x_train_scaled = ss.fit_transform(x_train)
x_test_scaled = ss.transform(x_test)

# Make predictions with the pre-trained model
y_xgb_train = xgb_model.predict(x_train_scaled)
y_xgb_test = xgb_model.predict(x_test_scaled)

# Compute metrics
cm_xgb_train = confusion_matrix(y_train, y_xgb_train)
cm_xgb_test = confusion_matrix(y_test, y_xgb_test)
a_xgb_train = accuracy_score(y_train, y_xgb_train)
a_xgb_test = accuracy_score(y_test, y_xgb_test)
f1_xgb_train = f1_score(y_train, y_xgb_train)
f1_xgb_test = f1_score(y_test, y_xgb_test)
r_xgb_train = recall_score(y_train, y_xgb_train)
r_xgb_test = recall_score(y_test, y_xgb_test)

# Display metrics
st.subheader('Model Performance Metrics')
st.write('Confusion Matrix for Training Data:')
st.write(cm_xgb_train)
st.write(f'Accuracy for Training Data: {a_xgb_train:.2f}')
st.write(f'F1-Score for Training Data: {f1_xgb_train:.2f}')
st.write(f'Recall for Training Data: {r_xgb_train:.2f}')
st.write('Confusion Matrix for Test Data:')
st.write(cm_xgb_test)
st.write(f'Accuracy for Test Data: {a_xgb_test:.2f}')
st.write(f'F1-Score for Test Data: {f1_xgb_test:.2f}')
st.write(f'Recall for Test Data: {r_xgb_test:.2f}')

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

# Display recall score for the user's input
# Note: Recall score for a single prediction does not make sense, so it's omitted here.
