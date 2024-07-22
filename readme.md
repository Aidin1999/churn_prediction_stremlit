# Churn Prediction Project

## Overview

The Churn Prediction Project aims to predict whether a customer is likely to leave (or "churn") from a bank using machine learning algorithms. The project includes the following key components:

- **Data Preparation**: The dataset comprises various customer features such as credit score, tenure, balance, and more. This data is used to train the models.
- **Model Training**: Several machine learning models are employed, including XGBoost and an Artificial Neural Network (ANN). These models are trained to predict customer churn based on the features provided.
- **Model Evaluation**: The performance of the models is evaluated using metrics such as recall to ensure that the models effectively identify customers who are likely to churn.
- **Deployment**: A Streamlit web application is developed to provide a user-friendly interface for predicting customer churn using a pre-trained XGBoost model.

## Files

### 1. `streamlit.py`

This Python file contains the Streamlit web application code. It allows users to:
- Input various customer features such as Credit Score, Tenure, Balance, etc.
- Get predictions on whether the customer is likely to leave the bank based on the pre-trained XGBoost model.
- View the confusion matrix and recall scores for the model.

### 2. `churnp.ipynb`

This Jupyter Notebook includes:
- **Data Processing**: Preparation and cleaning of the data.
- **Model Training**: Training of various machine learning models, including XGBoost and ANN.
- **Model Evaluation**: Evaluation of model performance using metrics like recall and confusion matrix.

### 3. `xgb_model.xgb`

This file contains the saved XGBoost model, which has been trained on the customer data. It is used by the Streamlit application to make predictions on new customer data.

## Important Notes

- **Data Access**: Unfortunately, the dataset used in this project cannot be shared due to privacy and confidentiality reasons. However, you can review the code to understand the data processing and model training steps.
- **Code Access**: The project code is available in the files mentioned above. You can run the Streamlit application and Jupyter Notebook to explore the functionality and performance of the models.

## Data Source

This model is based on data provided by Maktabkhooneh. The project is one of the final projects I have completed to obtain a certificate in machine learning from Maktabkhooneh.

## Getting Started

To get started with the project:
1. **Run the Streamlit Application**: Execute `p2_streamlit.py` to launch the web application. Ensure that you have the necessary libraries installed.
2. **Explore the Jupyter Notebook**: Review `churnp.ipynb` for details on data preparation, model training, and evaluation.
3. **Load the Model**: Use `xgb_model.xgb` in conjunction with the Streamlit application to make predictions.

For further assistance or questions, feel free to reach out!

---

Feel free to adjust or add any details based on your project's specific requirements or additional information you may want to include.

**Project by:** Aidin Miralmasi

**Certificate:** Matbakhonneh
