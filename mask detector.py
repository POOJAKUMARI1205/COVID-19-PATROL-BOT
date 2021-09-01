import tensorflow
from tensorflow.python.keras.models import load_model
import cv2 as cv
import numpy as np

model = load_model('model-017.model')

face_clsfr=cv.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}

while (True):

    ret, img = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_clsfr.detectMultiScale(img, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y + w, x:x + w]
        resized = cv.resize(face_img, (100, 100))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 1))
        result = model.predict(reshaped)
        print(result)
        label = np.argmax(result, axis=1)[0]

        cv.rectangle(img, (x, y), (x + w, y + h), color_dict[label], 2)
        cv.rectangle(img, (x, y - 40), (x + w, y), color_dict[label], -1)
        cv.putText(img, labels_dict[label], (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv.imshow('LIVE', img)
    key = cv.waitKey(1)

    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()
cap.release()