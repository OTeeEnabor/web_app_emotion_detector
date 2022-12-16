import streamlit as st
from processing import face_detector
# define the settings for the app

# define the header
st.title('Face Emotion Detector using a VGG16 CNN Model')

# define the information required to use the web app

# define image input
label = 'Upload your face for the model to detect your emotion.'
image_input = st.camera_input(label, key=None, help=None, on_change=None, args=None, kwargs=None,
                disabled=False, label_visibility="visible")
# run if statement to check if the image as been uploaded
if image_input is not None:
    st.write('Image uploaded')