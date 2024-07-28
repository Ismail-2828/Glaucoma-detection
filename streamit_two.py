import streamlit as st
from datetime import datetime
import pandas as pd
import os
import tempfile
import base64
import numpy as np
import cv2
from keras.models import load_model
from keras.applications.densenet import preprocess_input


model_path = r"model/model_final3_densenet.h5"
model = load_model(model_path)

def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded_image = base64.b64encode(image.read()).decode()
        background_css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
            font-weight: bold; 
        }}
        .stApp p {{
            font-weight: bold; 
        }}
        .stApp .streamlit-expanderHeader {{
            font-weight: bold;
        }}
        </style>
        """
        st.markdown(background_css, unsafe_allow_html=True)

def password_check():
    if "password_entered" not in st.session_state:
        st.session_state["password_entered"] = False
    animated_text = """
        <style>
        .welcome-text {
            font-size: 30px;
            font-weight: bold;
            color:blue;
            text-align: center;
            animation: fadeIn 2s infinite;
        }

        @keyframes fadeIn {
            0% { opacity: 0; }
            50% { opacity: 1; }
            100% { opacity: 0; }
        }
        </style>
        <div class="welcome-text">WELCOME TO ME-CURE DIAGNOSTIC CENTER</div>
        """
    st.markdown(animated_text, unsafe_allow_html=True)  
    password = st.text_input("Enter Password", type="password", key="unique_password_input")

    if st.button("Submit", key="unique_submit_button"):
        if password == "12345":  
            st.session_state["password_entered"] = True
            st.session_state["password_correct"] = True
            st.success("Password Correct")
        else:
            st.session_state["password_correct"] = False
            st.error("Password Incorrect")

def home():
    set_background("eye_check.jpg") 
    st.title("Glaucoma Information")

    menu = ["About Glaucoma", "Causes", "Risk Factor"]
    choice = st.radio("Select a topic:", menu, horizontal=True)

    if choice == "About Glaucoma":
        st.header("What Is Glaucoma?")
        st.write("""
        Glaucoma is a condition that damages your eye's optic nerve, and it gets worse over time. It's often linked to a buildup of pressure inside your eye. Glaucoma tends to run in families. You usually don’t get it until later in life.

        The increased pressure in your eye, called intraocular pressure, can damage your optic nerve that sends images to your brain. If the damage worsens, glaucoma can cause permanent vision loss or even total blindness within a few years.
        """)

    elif choice == "Causes":
        st.header("Glaucoma Causes")
        st.write("""
        The fluid inside your eye, called aqueous humor, usually flows out of your eye through a mesh-like channel. If this channel gets blocked or the eye is producing too much fluid, the liquid builds up. Sometimes, experts don’t know what causes this blockage, but it can be inherited, meaning it’s passed from parents to children.

        Less-common causes of glaucoma include a blunt or chemical injury to your eye, severe eye infection, blocked blood vessels inside your eye, and inflammatory conditions. It’s rare, but eye surgery to correct another condition can sometimes bring it on. It usually affects both eyes, but it may be worse in one than the other.
        """)

    elif choice == "Risk Factor":
        st.header("Glaucoma Risk Factors")
        st.write("""
        Glaucoma mostly affects adults older than 40, but young adults, children, and even infants can have it. African Americans tend to get it more often, at a younger age, and with more vision loss.
        """)

def save_patient_data(data, file_path):
    if not os.path.exists(file_path):
        data.to_csv(file_path, index=False)
    else:
        existing_data = pd.read_csv(file_path)
        updated_data = pd.concat([existing_data, data])
        updated_data.to_csv(file_path, index=False)

def load_patient_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame(columns=["Timestamp", "Name", "Age", "Family History", "Symptoms", "Prediction", "Confidence_Score"])

if "directory" not in st.session_state:
    st.session_state["directory"] = tempfile.mkdtemp()

file_path = os.path.join(st.session_state["directory"], "patient_data.csv")

def patient_info():
    st.title("Patient Information")
    st.write("Enter Patient Details")

    patient_name = st.text_input("Patient Name", key="unique_patient_name_input")
    age = st.number_input("Age", min_value=0, max_value=120, step=1, key="unique_age_input")
    last_eye_check = st.date_input("Last Eye Check", key="unique_last_eye_check_input")
    family_history = st.text_area("Family Disorder", placeholder="e.g., Glaucoma, Diabetes, Hypertension", key="unique_family_history_input")
    symptoms = st.text_area("Symptoms", placeholder="e.g., Blurred vision, Eye pain, Headaches", key="unique_symptoms_input")

    if st.button("Submit Information", key="unique_submit_info_button"):
        st.write("### Submitted Information")
        st.write(f"**Name:** {patient_name}")
        st.write(f"**Age:** {age}")
        st.write(f"**Last Eye Check:** {last_eye_check}")
        st.write(f"**Family History:** {family_history}")
        st.write(f"**Symptoms:** {symptoms}")

        data = {
            "Timestamp": [datetime.now()],
            "Name": [patient_name],
            "Age": [age],
            "Last Eye Check": [last_eye_check],
            "Family History": [family_history],
            "Symptoms": [symptoms],
            "Prediction": [""],   
            "Confidence_Score": [""] 
        }
        df = pd.DataFrame(data)

        try:
            save_patient_data(df, file_path)
            st.session_state["patient_info_submitted"] = True
            st.success("Information submitted successfully.")
        except Exception as e:
            st.error(f"An error occurred while writing to the file: {e}")

def results():
    if not st.session_state.get("patient_info_submitted", False):
        st.warning("Please return and complete the patient form!")
        return
    
    def preprocess_image(image_path):
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            image = cv2.resize(image, (224, 224))
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
        return image

    def predict(image_path):
        processed_image = preprocess_image(image_path)
        prediction = model.predict(processed_image)
        return prediction, processed_image
    
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
            .centered-content {
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .image-container {
                display: flex;
                justify-content: center;
            }
            .button-container {
                display: flex;
                justify-content: space-around;
                width: 50%;
            }
            .custom-button {
                padding: 10px;
                color: white;
                background-color: blue;
                border: none;
                border-radius: 5px;
                text-align: center;
                text-decoration: none;
                margin: 10px;
                flex: 1;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<h1 class="centered-title">Glaucoma Detection System</h1>', unsafe_allow_html=True)
        st.write("<div class='custom-header'><i>Upload an image of the eye for glaucoma testing.</i></div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        

        if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_path = temp_file.name

                image = cv2.imread(temp_path)
                st.markdown('<div class="image-container">', unsafe_allow_html=True)  
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Uploaded Image.', use_column_width=False)
                st.markdown('</div>', unsafe_allow_html=True)  


                if st.button('Predict'):
                    st.write("Predicting...")
                    prediction, processed_image = predict(temp_path)
                    prediction_label = "Glaucoma" if prediction[0] > 0.5 else "Normal"
                    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
                    st.write('<div class="button-container">', unsafe_allow_html=True)
                    st.markdown(f'<div class="custom-button">PREDICTION | {prediction_label}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="custom-button">CONFIDENCE | {confidence:.2f}</div>', unsafe_allow_html=True)
                    st.write('</div>', unsafe_allow_html=True)

                    if prediction_label == "Normal":
                        st.markdown('<h1 class="Normal-title" style="color: blue;">Congratulations, No Glaucoma detected!</h1>', unsafe_allow_html=True)
                    else:
                        st.markdown('<h1 class="Glaucoma-title" style="color: red;">Glaucoma detected, Consult Opthalmologist!</h1>', unsafe_allow_html=True)
                    
                    if os.path.isfile(file_path):
                        df = load_patient_data(file_path)
                        last_entry_index = df.index[-1]
                        df.loc[last_entry_index, "Prediction"] = prediction_label
                        df.loc[last_entry_index, "Confidence_Score"] = f"{confidence * 100:.2f}"
                        df.to_csv(file_path, index=False)

                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        if st.button('Download Patient Data'):
            patient_data = load_patient_data(file_path)
            csv = patient_data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  
            href = f'<a href="data:file/csv;base64,{b64}" download="patient_data.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
    main()

if "password_entered" not in st.session_state:
    st.session_state["password_entered"] = False
    
if "patient_info_submitted" not in st.session_state:
    st.session_state["patient_info_submitted"] = False

if not st.session_state["password_entered"]:
    password_check()
else:
    st.sidebar.title("Navigation")
    menu = ["Home", "Patient Form", "Diagnosis"]
    choice = st.sidebar.radio("Go to", menu)

    if choice == "Home":
        home()
    elif choice == "Patient Form":
        patient_info()
    elif choice == "Diagnosis":
        results()