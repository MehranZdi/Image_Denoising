from helper import predictor
import streamlit as st
import os
import cv2
import seaborn as sns
from PIL import Image
import tensorflow as tf
import numpy as np



def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('/home/mehranz/PycharmProjects/Denoising_streamlit/static/images',
                               uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1

    except:
        return 0

if __name__ == '__main__':
    sns.set_theme(style="darkgrid")
    sns.set()
    st.title('Image Denoiser')
    uploaded_file = st.file_uploader("Upload Image")


    if uploaded_file is not None:

        if save_uploaded_file(uploaded_file):
            uploaded_image_path = os.path.join(
                '/home/mehranz/PycharmProjects/Denoising_streamlit/static/images',
                uploaded_file.name)

            image = Image.open(uploaded_file)
            image = np.array(image)
            image = cv2.resize(image, (512, 512))
            st.image(image)
            predicted_image = predictor(uploaded_image_path)
            st.text('Predicted image: ')
            st.image(predicted_image, channels='RGB')