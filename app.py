import streamlit as st
import pandas as pd 
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('ohe_and_scaler.pkl','rb') as file:
    ohe_and_scaler = pickle.load(file)

# Split OHE from ohe_and_scaler variable
ohe_geo = ohe_and_scaler.named_transformers_['OneHotEncode']
ohe_categories = ohe_geo.categories_[0]
## Streamlit app
st.title('Customer Churn Prediction')
# User input
credit_score = st.number_input('Credit Score')
geography =st.selectbox('Geography',ohe_categories)
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.number_input('Age',18,92)
balance = st.number_input('Balance')
tenure = st.slider('Tenure',0,10)
num_of_products =st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit  Card',[0,1])
is_active_number =st.selectbox('Is Active Number',[0,1])
estimated_salary = st.number_input('Estimated Salary')

# Prepare the input data
if st.button('Predict'):
    input_data = pd.DataFrame({
        'CreditScore':[credit_score],
        'Geography':[geography],
        'Gender':[gender],
        'Age':[age],
        'Balance':[balance],
        'Tenure':[tenure],
        'NumOfProducts':[num_of_products],
        'HasCrCard':[has_cr_card],
        'IsActiveMember':[is_active_number],
        'EstimatedSalary': [estimated_salary]
    })

# Feature engineering
    input_data_df = pd.DataFrame(input_data)
    input_data_df['Gender'] = label_encoder_gender.transform(input_data_df['Gender'])
    input_data_ohe_and_scaler = ohe_and_scaler.transform(input_data_df)

# Prediction
    def Prediction(input_data):
        prediction = model.predict(input_data)
        prob_of_prediction = prediction[0][0]
        st.write(f'Probability is {prob_of_prediction:2f}')
        if prob_of_prediction >0.5:
            st.write(f'The customer is likely to churn.')
        else:
            st.write(f'The customer is not likely to churn.')

# Call the Prediction function
    Prediction(input_data_ohe_and_scaler)




