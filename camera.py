import cv2 as cv
import numpy as np
import pickle
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.backend import set_session
import tensorflow as tf

from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

sess = tf.Session()
graph = tf.get_default_graph()

# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras!
# Otherwise, their weights will be unavailable in the threads after the session there has been set
set_session(sess)

model = load_model("model-017.model")

face_clsfr=cv.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)

labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}

class VideoCamera(object):
    def __init__(self):
        # capturing video
        self.video = cv.VideoCapture(0)
    def __del__(self):
        # releasing camera
        self.video.release()
    def get_frame(self):
        # extracting frames
        ret, img = self.video.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_clsfr.detectMultiScale(img, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y + w, x:x + w]
            resized = cv.resize(face_img, (100, 100))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 100, 100, 1))
            global sess
            global graph
            with graph.as_default():
                set_session(sess)
                result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]
            cv.rectangle(img, (x, y), (x + w, y + h), color_dict[label], 2)
            cv.rectangle(img, (x, y - 40), (x + w, y), color_dict[label], -1)
            cv.putText(img, labels_dict[label], (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        ret,jp = cv.imencode('.jpg',img)
        return jp.tobytes()