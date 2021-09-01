import cv2 as cv
import numpy as np
import pickle
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.backend import set_session
import tensorflow as tf

from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

sess2 = tf.Session()
graph2 = tf.get_default_graph()

# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras!
# Otherwise, their weights will be unavailable in the threads after the session there has been set
set_session(sess2)

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)


class VideoCamera2(object):
    def __init__(self):
        # capturing video
        self.video = cv.VideoCapture(0)
    def __del__(self):
        # releasing camera
        self.video.release()
    def get_frame(self):
        # extracting frames
        ret, img = self.video.read()
        # Convert to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        ret,jp = cv.imencode('.jpg',img)
        return jp.tobytes()