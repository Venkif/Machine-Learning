# 1 is good (lower risk) and 0 is bad (Higher risk)
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("Xgb_credit_model.pkl")
encoders ={col: joblib.load(f"{col}_encoder.pkl") for col in ["Sex","Housing","Saving accounts","Checking account"]}

st.title("Credit Risk Prediction App")
st.write("Enter applicant information to predict if the credit risk is Good or Bad")

age = st.number_input("Age", min_value=18, max_value=80,value=20)
sex = st.selectbox("Sex", ["male","female"])
job = st.number_input("Job (0-3)",min_value=0,max_value=3,value=1)
housing = st.selectbox("Housing", ["own", "rent", "free"])
saving_accounts = st.selectbox("Saving account",["little","moderate","rich","quite Rich"])
Checking_accounts = st.selectbox("Checking account",["little","moderate","rich"])
credit_amount = st.number_input("Credit amount",min_value=0,value=1000)
duration = st.number_input("Duration (months)",min_value=1,value=1)

input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [job],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account": [encoders["Checking account"].transform([Checking_accounts])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration]
})

st.write("Input data used for prediction:")
st.dataframe(input_df)

if st.button("Predict Risk"):
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.success("The predicted credit risk is 'GOOD'")
    else:
        st.error("The predicted credit risk is 'BAD'")
