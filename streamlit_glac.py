import streamlit as st
import numpy as np
import cv2
from PIL import Image
from keras.models import load_model
from keras.applications.densenet import preprocess_input
import tempfile
import os

model_path = r"model/model_final_densenet.h5"
model = load_model(model_path)

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

def main():
    st.title('Glaucoma Detection')
    st.write("Upload an image of the eye for glaucoma detection.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=False)

        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file, format='JPEG')
            temp_path = temp_file.name
        
        if st.button('Predict'):
            st.write("Predicting...")
            prediction = predict(temp_path)
            prediction_label = "Glaucoma" if prediction[0] > 0.5 else "Normal"
            st.write(f'Prediction: {prediction_label}')
            st.write(f'Confidence: {prediction[0][0]:.2f}' if prediction[0][0] > 0.5 else f'Confidence: {1 - prediction[0][0]:.2f}')

        
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    main()