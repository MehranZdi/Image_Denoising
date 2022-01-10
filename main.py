from helper import *
import streamlit as st
import os
import seaborn as sns
from PIL import Image

###   Frontend part:

# [theme]
primaryColor="#F63366"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="sans serif"
sns.set_theme(style="darkgrid")
sns.set()
st.title('Image Denoiser')


def save_uploaded_file(uploaded_file):

    try:
        with open(os.path.join('static/images',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1

    except:
        return 0


test_path = '/home/mehranz/Documents/Datasets/Denoising_face/FileMakerModule_images'
uploaded_file = st.file_uploader("Upload Image")

# text over upload button "Upload Image"

if uploaded_file is not None:

    if save_uploaded_file(uploaded_file):

        # display the image

        display_image = Image.open("/home/mehranz/Downloads/Admin's photo.jpg")
        st.image(display_image)
        prediction = predictor(test_path, os.path.join('static/images', uploaded_file.name))
        print('DDDDDDDDDDDDDDDooooooooooooooooNNNNNNNNNNNNNNNNeeeeeeeeeeeeeeee')
        st.image(prediction[-1])
        os.remove('static/images/' + uploaded_file.name)
        # deleting uploaded saved picture after prediction
        # drawing graphs
        st.text('Predictions :-')
        # fig, ax = plt.subplots()
        # ax  = sns.barplot(y='name', x='values', data=prediction, order=prediction.sort_values('values', ascending=False).name)
        # ax.set(xlabel='Confidence %', ylabel='Breed')
        # st.pyplot(fig)