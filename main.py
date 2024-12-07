from turtle import color
import streamlit as st
import tensorflow as tf
import numpy as np

import google.generativeai as genai
import os


genai.configure(api_key="put_your_gemini_flash_api_key_here")
chat_model = genai.GenerativeModel("gemini-1.5-flash")


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition","List of Plants"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home.jpg"
    st.image(image_path,use_container_width=True)
    st.markdown("""
    ### Welcome to the Plant Disease Recognition System!
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Check:** **List of Plants**
    2. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    3. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    4. **Chatbot:** Our chatbot will help you to know more about the disease by giving you the symptoms of the disease and how to prevent it.Also you can ask for any query.         
    5. **Results:** View the results and disease bot help for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.
    - **Disease Help Bot:** Helps the user for further action. 

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                The Plant Disease Recognition System enhances agricultural practices by providing timely information and support. The integration of a chatbot facilitates user engagement, making the system more efficient and user-friendly.
                ### Contact us:
                **Email:** plantdisease@gmail.com 

                **Phone:** 01642005823
                """)
    

elif(app_mode=="List of Plants"):
    st.header("List of Plants")
    items = ["Potato", "Grape","Corn", "Tomato", "Apple","Raspberry","Orange","Peach","Cherry","Bluberry","Strawberry","Soybean","Bell_Pepper"]
    for item in items:
        st.text(item)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not 'ClASS_PREDICTION' in st.session_state:
        st.session_state.ClASS_PREDICTION = []

    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image") and test_image):
        st.image(test_image,width=4,use_container_width=True)
    
    #Predict button
    if(st.button("Predict")):
        with st.status("Predicting Plant Disease", expanded=True) as status:
            result_index = model_prediction(test_image)
            #Reading Labels
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy']
            st.session_state.ClASS_PREDICTION.append(class_name[result_index])
            response = chat_model.generate_content(f"Please Explain this plant disease  (In Bangla Language): {st.session_state.ClASS_PREDICTION[-1]}")
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            st.rerun()
        
    if st.session_state.ClASS_PREDICTION:
        st.success("Model is Predicting it's a {}".format(st.session_state.ClASS_PREDICTION[-1]))
        with st.container(height=600):
            st.title("Disease Help Bot")
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        prompt = st.chat_input("Ask more about the diagnosed disease")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            response = chat_model.generate_content(f"Please give your answers (in Bangla Language) based on the context of this plant disease {st.session_state.ClASS_PREDICTION[-1]}. Prompt: {prompt}")
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            st.rerun()

    else:
        st.warning("UPLOAD A IMAGE AND CLICK 'PREDICT'")

            
