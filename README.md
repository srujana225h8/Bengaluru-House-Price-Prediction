# **Bengaluru-House-Price-Prediction**

This project predicts house prices in Bengaluru using Machine Learning. The model is trained on housing data and deployed using Streamlit so users can input property details and get an estimated price.

The application demonstrates the complete ML workflow including data preprocessing, model training, model serialization, and deployment through a simple web interface.

## Project Overview

The goal of this project is to build a machine learning model that predicts house prices based on features such as:

• Location  
• Total Square Feet  
• Number of Bedrooms (BHK)  
• Number of Bathrooms  

The trained model is integrated with a Streamlit application to provide an interactive prediction interface.

## Project Structure

```
Bengaluru-House-Price-Prediction
│
├── app.py
├── Linear_model.pkl
├── encoder.pkl
├── requirements.txt
├── README.md
└── dataset.csv
```
app.py              → Streamlit web application  
Linear_model.pkl    → Trained machine learning model  
encoder.pkl         → Encoded location data  
requirements.txt    → Required Python libraries  
README.md           → Project documentation  
dataset.csv         → Dataset used for training


## Software and Tools Requirements
1. [GitHub](https://github.com)
2. [StreamlitCloudAccount](https://streamlit.io/cloud)
3. [VSCode](https://code.visualstudio.com/download)

## Create Virtual Environment

```
python -m venv venv
```
## activate virtual environment

```
venv\Scripts\activate
```

## Project setup

```
pip install -r requirements.txt
```
## Website Link

1. [Bengaluru RealEstate Prediction🏡](https://bengaluru-house-price-prediction-nbf4ub62uy3cdg9gsdgtth.streamlit.app/)


