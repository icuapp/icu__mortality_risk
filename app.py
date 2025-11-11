import streamlit as st
import pandas as pd
import pickle

st.title('ICU Mortality Prediction App')

# Load model and preprocessor
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('preprocessor.pkl', 'rb') as pre_file:
    preprocessor = pickle.load(pre_file)

# User inputs
age = st.number_input('Age', 0, 120)
blood_pressure = st.number_input('Blood Pressure', 0, 300)
heart_rate = st.number_input('Heart Rate', 0, 200)
gender = st.selectbox('Gender', ['M', 'F'])
icu_type = st.selectbox('ICU Type', ['cardiac', 'neuro', 'others'])

# On predict button click
if st.button('Predict'):
    input_df = pd.DataFrame([{
        'age': age,
        'blood_pressure': blood_pressure,
        'heart_rate': heart_rate,
        'gender': gender,
        'icu_type': icu_type
    }])
    X = preprocessor.transform(input_df)
    prediction = model.predict(X)
    st.write('Predicted Mortality:', prediction[0])
