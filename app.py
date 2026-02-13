import streamlit as st
import pickle
import pandas as pd

# Load model and encoders
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

st.title("💻 Laptop Price Prediction App")

st.write("Enter laptop details below:")

brand = st.selectbox("Brand", encoders["brand"].classes_)
ram = st.selectbox("RAM (GB)", [4, 8, 16, 32])
storage = st.selectbox("Storage (GB)", [256, 512, 1024])
processor = st.selectbox("Processor", encoders["processor"].classes_)

if st.button("Predict Price"):
    brand_encoded = encoders["brand"].transform([brand])[0]
    processor_encoded = encoders["processor"].transform([processor])[0]

    input_data = pd.DataFrame([[brand_encoded, ram, storage, processor_encoded]],
                              columns=["brand", "ram", "storage", "processor"])

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Price: ₹ {round(prediction, 2)}")


