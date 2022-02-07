import tensorflow as tf
import cv2
from skimage.util import random_noise
import numpy as np

model = tf.keras.models.load_model(
        r'/home/mehranz/PycharmProjects/Denoising_streamlit/static/DenoiserModel.h5')


def predictor(uploaded_image_path):

    uploaded_image = cv2.imread(uploaded_image_path)
    uploaded_image = cv2.resize(uploaded_image, (512, 512))
    uploaded_image = random_noise(uploaded_image, mode='s&p', amount=0)
    uploaded_image = tf.expand_dims(uploaded_image, axis=0)
    predicted = model.predict(uploaded_image)
    predicted = tf.squeeze(predicted, axis=0)
    predicted = np.array(predicted)
    predicted = cv2.cvtColor(predicted, cv2.COLOR_BGR2RGB)
    predicted = np.array(predicted)
    pil_img = tf.keras.preprocessing.image.array_to_img(predicted)

    return predicted