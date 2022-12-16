import numpy as np
import PIL
import cv2
import tensorflow
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def face_detector(image):
    """

    :param image (None or UploadFile): The UploadedFile class is a subclass of BytesIO, and
    therefore it is "file-like". This means you can pass them anywhere where a file is expected.

    :return:
    """
    bytes_data = image.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.unit8), cv2.Color_BGR2GRAY)
    gray = cv2.cvtColor(img, cv2.Color_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    try:
        face = faces[0]
        (x, y, w, h) = face
        roi_color = img[y:y+h, x:x+w]
        resized = cv2.resize(roi_color, (224,224))
        # return resized image
        return resized
    except Exception as e:
        print(e)
        print("No faces detected")

def make_prediction(resized_image, model):
    """

    :param resized_image:
    :param model:
    :return:
    """
    return list_values