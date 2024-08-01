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
        <div class="welcome-text">WELCOME TO VICBOR DIAGNOSTIC CENTER</div>
        """
    st.markdown(animated_text, unsafe_allow_html=True)  
   
    data = {
    "VBH/OD/705": "Doctor@12",
    "VBH/OD/743": "Nurse@12",
    "VBH/OD/547": "Health@12",
    }

    user_name = st.text_input("Username",key="username_key")
    pass_word = st.text_input("Password",type="password", key="password_key")

    submit_button = st.button('Submit',key="Submit_button_key")

    for Key,val in data.items():
        if submit_button:
            if user_name == Key and pass_word == val:
                st.session_state["password_entered"] = True
                st.session_state["password_correct"] = True
                st.success("Password Correct")
                break
            else:
                st.session_state["password_correct"] = False
                st.error("Password Incorrect")
                break

def home():
    set_background("eye_check.jpg") 
    st.markdown(
            """
            <style>
            .glaucoma-title {
                text-align: center;
                font-weight: bold;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    st.markdown('<h1 class="glaucoma-title">GLAUCOMA</h1>', unsafe_allow_html=True)
    st.session_state["patient_info_submitted"] = False 
    menu = ["About Glaucoma", "Risk Factor", "Treatment"]
    choice = st.radio("Select a topic:", menu, horizontal=True)
    
    st.markdown("""
        <style>
        .stRadio > div {
            display: flex;
            flex-direction: row;
        }
        .stRadio > div > label {
            margin-right: 110px; 
        }
        </style>
    """, unsafe_allow_html=True)

    if choice == "About Glaucoma":
        st.header("What Is Glaucoma?")
        st.markdown("""
            <style>
            .justified-text {
                text-align: justify;
                font-weight: bold;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="justified-text">
                Glaucoma, a leading cause of blindness in the world, is a group of ocular 
                pathologies which progressively damages the optic nerve of the eye, the 
                nerve that transmit visual impulses to and from the brain. 
                It is a group of degenerative eye disorders typically associated with 
                increase in the intraocular pressure  against the eye walls, thereby 
                damaging the optic nerve head and affecting the visual field.
            </div>
        """, unsafe_allow_html=True)
        
        st.header("Types of Glaucoma")
        st.markdown("""
            <style>
            .justified-text-types {
                text-align: justify;
                font-weight: bold;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="justified-text-types">
                The two types of glaucoma are the primary open-angle glaucoma (POAG) 
                and the angle closure glaucoma. While both pose serious threat on 
                ocular health, the primary open-angle glaucoma is often asymptomatic 
                in its early stage, until severe, more significant and often times 
                irreversible damage is done to the eye. The angle closure glaucoma, 
                on the other hand can be sudden, more severe and painful, as as a 
                result of rapid surge or increase in intraocular pressure of the eye.
                The problem of glaucoma is one of the most concerning issues in public health.
            </div>
        """, unsafe_allow_html=True)
        
    elif choice == "Risk Factor":
        st.header("Glaucoma Risk Factors")
        st.write("""
        Glaucoma mostly affects adults older than 40, but young adults, children, and even infants can have it. African Americans tend to get it more often, at a younger age, and with more vision loss.
        Common Risk Factors include:

         Black Ancestry

         Age ≥ 60yrs

         Family Record of Glaucoma or other ocular conditions

         Diabetic Patient

         Cornea being thinner than normal 

         High Blood Pressure or Sickle Cell Anaemia
        
        """)
        
    elif choice == "Treatment":
        st.header("Treatment")
        st.write("""
        Possible therapeutic remedies may include:
        
        Medicinal therapy
        
        Laser therapy
        
        Surgical interventions
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
    family_history = st.text_area("Family Disorder, if any", placeholder="e.g., Glaucoma, Diabetes, Hypertension", key="unique_family_history_input")
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
            .norm-container {
                display: flex;
                justify-content: center;
                margin-top: 20px;
            }
            .glac-container {
                display: flex;
                justify-content: center;
                margin-top: 20px;
            }
            .Normal-unclickable-button {
                display: inline-block;
                padding: 10px 20px;
                font-size: 20px;
                color: white;
                background-color: blue;
                border: none;
                border-radius: 5px;
                text-align: center;
                text-decoration: none;
                cursor: default;
                margin-top: 20px;
            }
            .Glaucoma-unclickable-button {
                display: inline-block;
                padding: 10px 20px;
                font-size: 20px;
                color: white;
                background-color: red;
                border: none;
                border-radius: 5px;
                text-align: center;
                text-decoration: none;
                cursor: default;
                margin-top: 20px;
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
                        st.markdown("""
                            <div class="norm-container">
                                <div class="Normal-unclickable-button">Congratulations, No Glaucoma detected!</div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class="glac-container">
                                <div class="Glaucoma-unclickable-button">Glaucoma detected, consult opthalmologist!</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
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
        
        
import streamlit as st
from datetime import datetime
import pandas as pd
import os
import base64
import numpy as np
import cv2
from keras.models import load_model
from keras.applications.densenet import preprocess_input
import tempfile

model_path = r"C:\Users\ismai\Downloads\model_final3_densenet.h5"
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
    # password = st.text_input("Enter Password", type="password", key="unique_password_input")
    
    
    data = {
    "Emmy@75": 1234,
    "Moshood@15": 5678
    }

    user_name = st.text_input("Username",key="username_key")
    pass_word = st.text_input("Password",key="password_key")

    submit_button = st.button('Submit',key="Submit_button_key")

    for Key,val in data.items():
        if submit_button:
            if user_name == Key and pass_word == str(val):
                st.session_state["password_entered"] = True
                st.session_state["password_correct"] = True
                st.success("Password Correct")
                break
            else:
                st.session_state["password_correct"] = False
                st.error("Password Incorrect")


def home():
    set_background(r"C:\Users\ismai\Downloads\small_scale_waste_converter\eye_check.jpg") 
    st.markdown(
            """
            <style>
            .glaucoma-title {
                text-align: center;
                font-weight: bold;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    st.markdown('<h1 class="glaucoma-title">GLAUCOMA</h1>', unsafe_allow_html=True)
    st.session_state["patient_info_submitted"] = False  
    menu = ["About Glaucoma", "Risk Factor", "Treatment"]
    choice = st.radio("Select a topic:", menu, horizontal=True)
    
    st.markdown("""
        <style>
        .stRadio > div {
            display: flex;
            flex-direction: row;
        }
        .stRadio > div > label {
            margin-right: 110px; 
        }
        </style>
    """, unsafe_allow_html=True)

    if choice == "About Glaucoma":
        st.header("What Is Glaucoma?")
        st.markdown("""
            <style>
            .justified-text {
                text-align: justify;
                font-weight: bold;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="justified-text">
                Glaucoma, a leading cause of blindness in the world, is a group of ocular 
                pathologies which progressively damages the optic nerve of the eye, the 
                nerve that transmit visual impulses to and from the brain. 
                It is a group of degenerative eye disorders typically associated with 
                increase in the intraocular pressure  against the eye walls, thereby 
                damaging the optic nerve head and affecting the visual field.
            </div>
        """, unsafe_allow_html=True)
        
        st.header("Types of Glaucoma")
        st.markdown("""
            <style>
            .justified-text-types {
                text-align: justify;
                font-weight: bold;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="justified-text-types">
                The two types of glaucoma are the primary open-angle glaucoma (POAG) 
                and the angle closure glaucoma. While both pose serious threat on 
                ocular health, the primary open-angle glaucoma is often asymptomatic 
                in its early stage, until severe, more significant and often times 
                irreversible damage is done to the eye. The angle closure glaucoma, 
                on the other hand can be sudden, more severe and painful, as as a 
                result of rapid surge or increase in intraocular pressure of the eye.
                The problem of glaucoma is one of the most concerning issues in public health.
            </div>
        """, unsafe_allow_html=True)
        
    elif choice == "Risk Factor":
        st.header("Glaucoma Risk Factors")
        st.write("""
        Glaucoma mostly affects adults older than 40, but young adults, children, and even infants can have it. African Americans tend to get it more often, at a younger age, and with more vision loss.
        Common Risk Factors include:

         Black Ancestry

         Age ≥ 60yrs

         Family Record of Glaucoma or other ocular conditions

         Diabetic Patient

         Cornea being thinner than normal 

         High Blood Pressure or Sickle Cell Anaemia
        
        """)
        
    elif choice == "Treatment":
        st.header("Treatment")
        st.write("""
        Possible therapeutic remedies may include:
        
        Medicinal therapy
        
        Laser therapy
        
        Surgical interventions
        """)

def patient_info():
    st.title("Patient Information")
    st.write("Enter Patient Details")

    patient_name = st.text_input("Patient Name", key="unique_patient_name_input")
    age = st.number_input("Age", min_value=0, max_value=120, step=1, key="unique_age_input")
    family_history = st.text_area("Family History", placeholder="e.g., Glaucoma, Diabetes, Hypertension", key="unique_family_history_input")
    symptoms = st.text_area("Symptoms", placeholder="e.g., Blurred vision, Eye pain, Headaches", key="unique_symptoms_input")

    if st.button("Submit Information", key="unique_submit_info_button"):
        st.write("### Submitted Information")
        st.write(f"**Name:** {patient_name}")
        st.write(f"**Age:** {age}")
        st.write(f"**Family History:** {family_history}")
        st.write(f"**Symptoms:** {symptoms}")

        data = {
            "Timestamp": [datetime.now()],
            "Name": [patient_name],
            "Age": [age],
            "Family History": [family_history],
            "Symptoms": [symptoms],
            "Prediction": [""],   
            "Confidence_Score": [""] 
        }
        df = pd.DataFrame(data)
        file_path = r"C:\Users\ismai\Desktop\Glaucoma_Project\file.csv"

        try:
            file_exists = os.path.isfile(file_path)
            df.to_csv(file_path, mode='a', index=False, header=not file_exists)
            st.session_state["patient_info_submitted"] = True
            st.success("Information submitted successfully.")
        except Exception as e:
            st.error(f"An error occurred while writing to the file: {e}")

def results():
    if not st.session_state.get("patient_info_submitted", False):
        st.warning("Go back and Fill the patient form.")
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
            .norm-container {
                display: flex;
                justify-content: center;
                margin-top: 20px;
            }
            .glac-container {
                display: flex;
                justify-content: center;
                margin-top: 20px;
            }
            .Normal-unclickable-button {
                display: inline-block;
                padding: 10px 20px;
                font-size: 20px;
                color: white;
                background-color: blue;
                border: none;
                border-radius: 5px;
                text-align: center;
                text-decoration: none;
                cursor: default;
                margin-top: 20px;
            }
            .Glaucoma-unclickable-button {
                display: inline-block;
                padding: 10px 20px;
                font-size: 20px;
                color: white;
                background-color: red;
                border: none;
                border-radius: 5px;
                text-align: center;
                text-decoration: none;
                cursor: default;
                margin-top: 20px;
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
            st.write('<div class="centered-content">', unsafe_allow_html=True)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Uploaded Image.', use_column_width=False)

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
                        st.markdown("""
                            <div class="norm-container">
                                <div class="Normal-unclickable-button">Congratulations, No Glaucoma detected!</div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                        st.markdown("""
                            <div class="glac-container">
                                <div class="Glaucoma-unclickable-button">Glaucoma detected, consult opthalmologist!</div>
                            </div>
                        """, unsafe_allow_html=True)
                
                file_path = r"C:\Users\ismai\Desktop\Glaucoma_Project\file.csv"
                if os.path.isfile(file_path):
                    df = pd.read_csv(file_path)
                    if not df.empty:
                        df.iloc[-1, df.columns.get_loc("Prediction")] = prediction_label
                        df.iloc[-1, df.columns.get_loc("Confidence_Score")] = confidence
                        df.to_csv(file_path, index=False)
        else:
            st.write('</div>', unsafe_allow_html=True)

    main()

if "password_entered" not in st.session_state:
    st.session_state["password_entered"] = False
if "patient_info_submitted" not in st.session_state:
    st.session_state["patient_info_submitted"] = False

if not st.session_state["password_entered"]:
    password_check()
else:
    pages = {
        "Home": home,
        "Patient Info": patient_info,
        "Results": results
    }

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    page = pages[selection]
    page()