import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from keras.applications.densenet import preprocess_input
import tempfile
import os

# Load the model
model_path = r"model/model_final_densenet.h5"
model = load_model(model_path)

# Define functions
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
    return image

def predict(image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    return prediction

# Main app
def main():
    st.markdown(
        """
        <style>
        .centered-title {
            text-align: center;
            border: 2px solid blue;
            padding: 6px;
            border-radius: 10px;
        }
        .custom-header {
            text-align: center;
        }
        .prediction-button {
            display: inline-block;
            padding: 10px;
            color: white;
            background-color: blue;
            border: none;
            border-radius: 5px;
            text-align: center;
            text-decoration: none;
            margin: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('<h1 class="centered-title">Glaucoma Detection System</h1>', unsafe_allow_html=True)
    st.write("<div class='custom-header'><i>Upload an image of the eye for glaucoma testing.</i></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns([2, 1])
        with col1:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name

            image = cv2.imread(temp_path)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Uploaded Image.', use_column_width=False)

            if st.button('Predict'):
                st.write("Predicting...")
                prediction = predict(temp_path)
                prediction_label = "Glaucoma" if prediction[0] > 0.5 else "Normal"
                confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
                
                with col2:
                    st.markdown(f'<div class="prediction-button">PREDICTION | {prediction_label}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="prediction-button">CONFIDENCE | {confidence:.2f}</div>', unsafe_allow_html=True)

            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == '__main__':
    main()
