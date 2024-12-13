import streamlit as st
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np

# Path to the model and metrics
model_save_path = r"D:\movies_prediction\project\models"
model_file = os.path.join(model_save_path, "random_forest_model.pkl")
classification_report_image = os.path.join(model_save_path, "classification_report.png")
confusion_matrix_image = os.path.join(model_save_path, "confusion_matrix.png")
data_path = r"D:\movies_prediction\project\cleaned_movie_data.csv"

# Streamlit UI setup
st.title('Movie Genre Prediction')

# Load the saved model
@st.cache
def load_model():
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
    return model

# Load and display saved classification report image
def display_classification_report():
    st.subheader("Classification Report Heatmap")
    st.image(classification_report_image, caption='Classification Report Heatmap', use_column_width=True)

# Load and display saved confusion matrix image
def display_confusion_matrix():
    st.subheader("Confusion Matrix")
    st.image(confusion_matrix_image, caption='Confusion Matrix', use_column_width=True)

# Predict movie genre
def predict_movie_genre(model, input_data):
    # Example feature columns (adjust based on your dataset)
    X = pd.DataFrame([input_data], columns=["avg_rating", "total_ratings", "unique_users"])
    prediction = model.predict(X)
    return prediction[0]

# Streamlit interface
st.sidebar.header('Input Features')
avg_rating = st.sidebar.number_input("Average Rating", min_value=0.0, max_value=10.0, step=0.1)
total_ratings = st.sidebar.number_input("Total Ratings", min_value=0, step=1)
unique_users = st.sidebar.number_input("Unique Users", min_value=0, step=1)

# Predict button
if st.sidebar.button('Predict Genre'):
    # Load the trained model
    model = load_model()
    
    # Prepare input data
    input_data = [avg_rating, total_ratings, unique_users]
    
    # Predict the genre
    predicted_genre = predict_movie_genre(model, input_data)
    st.subheader(f"Predicted Genre: {predicted_genre}")

# Display classification report and confusion matrix
display_classification_report()
display_confusion_matrix()

