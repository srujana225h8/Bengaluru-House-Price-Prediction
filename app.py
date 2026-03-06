# streamlit UI
import pickle
import streamlit as st
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Bangalore House Price Predictor🏠",
    page_icon="house_logo.png",
    layout="wide"
)


# ---------------- LOAD MODEL ----------------
with open("Linear_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("label_encoder.pkl", "rb") as file1:
    encoder = pickle.load(file1)

df = pd.read_csv("cleaned_data.csv")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🏠 Bangalore House Price Prediction")
    st.image("house_logo.png", use_container_width=True)

    st.markdown("---")

    st.subheader("About")
    st.write(
        "This app predicts house prices in Bangalore using a trained Machine Learning model."
    )

    st.markdown("**Model Used:** Linear Regression")

    st.markdown("---")

    if st.checkbox("Show Dataset Sample"):
        st.write(df.head())

# ---------------- MAIN TITLE ----------------
st.title("🏡 Bangalore House Price Prediction App")
st.markdown("Enter property details to estimate the house price.")

st.markdown("---")

# ---------------- INPUT SECTION ----------------
st.subheader("Property Details")

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
    "Total Square Feet",
    min_value=300,
    max_value=10000,
    value=1200,
    step=50
)

# ---------------- PROPERTY SUMMARY ----------------
st.markdown("### Property Summary")

s1, s2, s3, s4 = st.columns(4)

s1.metric("Location", location)
s2.metric("BHK", bhk)
s3.metric("Bathrooms", bath)
s4.metric("Area (sqft)", sqft)

st.markdown("---")

# ---------------- ENCODE LOCATION ----------------
encoded_loc = encoder.transform([location])
new_data = [[bhk, sqft, bath, encoded_loc[0]]]

# ---------------- PREDICTION BUTTON ----------------
colA, colB, colC = st.columns([1, 2, 1])

if colB.button("💰 Predict House Price"):

    # loading progress bar
    progress = st.progress(0)

    for i in range(100):
        progress.progress(i + 1)

    pred = model.predict(new_data)[0]
    pred = round(pred * 100000)

    st.success("Prediction Generated Successfully")

    # result container
    with st.container():
        st.markdown("### 🏷 Predicted House Price")

        st.metric(
            label="Estimated Property Value",
            value=f"₹ {pred:,.0f}"
        )

        st.info("This value is an estimated market price based on the trained ML model.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Machine Learning Project | Bangalore House Price Prediction")