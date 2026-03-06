import pickle
import streamlit as st
import pandas as pd

# Application page configuration
st.set_page_config(
    page_title="Bangalore House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Loading trained model
with open("Linear_model.pkl", "rb") as file:
    model = pickle.load(file)

# Loading saved encoder
with open("label_encoder.pkl", "rb") as file1:
    encoder = pickle.load(file1)

# Reading the cleaned dataset
df = pd.read_csv("cleaned_data.csv")

# Sidebar layout and information
with st.sidebar:
    st.title("🏠 Bangalore House Price Prediction")
    st.image("house_logo.png", use_container_width=True)

    st.markdown("---")

    st.subheader("Application Info")
    st.write(
        "Provide the property details and the model will estimate the expected house price."
    )

    st.markdown("Model Type: Linear Regression")

    st.markdown("---")

    if st.checkbox("View Sample Data"):
        st.write(df.head())

# Main title of the application
st.title("🏡 Bangalore House Price Estimator")
st.write("Fill in the property information below to generate a predicted price.")

st.markdown("---")

# Section where users enter property information
st.subheader("Enter Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    location = st.selectbox(
        "Location",
        options=sorted(df["location"].unique())
    )

with col2:
    bhk = st.selectbox(
        "BHK",
        options=sorted(df["BHK"].unique())
    )

with col3:
    bath = st.selectbox(
        "Bathrooms",
        options=sorted(df["bath"].unique())
    )

sqft = st.slider(
    "Total Area (Square Feet)",
    min_value=300,
    max_value=10000,
    value=1200,
    step=50
)

# Displaying selected property details
st.markdown("### Selected Property Summary")

s1, s2, s3, s4 = st.columns(4)

s1.metric("Location", location)
s2.metric("BHK", bhk)
s3.metric("Bathrooms", bath)
s4.metric("Area (sqft)", sqft)

st.markdown("---")

# Encoding location value for prediction
encoded_loc = encoder.transform([location])

# Preparing the input structure for the model
new_data = [[bhk, sqft, bath, encoded_loc[0]]]

# Creating a centered prediction button
left, middle, right = st.columns([1, 2, 1])

with middle:
    predict = st.button("💰 Predict House Price", use_container_width=True)

# Generating prediction when button is clicked
if predict:
    with st.spinner("Calculating predicted value..."):
        pred = model.predict(new_data)[0]
        pred = round(pred * 100000)

    st.success("Prediction Generated")

    st.markdown("### 🏷 Estimated House Price")

    st.metric(
        label="Predicted Property Value",
        value=f"₹ {pred:,.0f}"
    )

    st.info("The value shown is an approximate estimate produced by the machine learning model.")

# Footer information
st.markdown("---")
st.caption("Machine Learning Project | Bangalore House Price Prediction")
