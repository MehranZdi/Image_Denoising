import tensorflow as tf
import cv2
import os
import FileMaker as fm


current_path = os.getcwd()
model = tf.keras.models.load_model(r'/home/mehranz/PycharmProjects/Denoising_streamlit/static/FaceUnetDenoising.h5')
test_path = '/home/mehranz/Documents/Datasets/Denoising_face/FileMakerModule_images'


def predictor(path, uploaded_image):

    files = fm.File_Maker(test_path)
    images = files.file_maker()
    dicti = files.list_to_dict(images)
    uploaded_image = cv2.resize(uploaded_image, (256, 256))
    dicti['uploaded'] = uploaded_image
    images.append(uploaded_image)
    predicted = model.predict(images)

    return predicted