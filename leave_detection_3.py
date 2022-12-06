import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import Model
import threading
import time
from keras.applications.vgg16 import preprocess_input

# load the model
tflite_model = tf.lite.Interpreter(model_path="resources/plant_diseas_model.tflite")

# tflite_model.resize_tensor_input(0, [-1, 224, 224, 3])
tflite_model.allocate_tensors()


# all the classes to be predicted by the model
CLASSES = [
    "Tomato___Late_blight",
    "Tomato___healthy",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Potato___healthy",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Tomato___Early_blight",
    "Tomato___Septoria_leaf_spot",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Strawberry___Leaf_scorch",
    "Peach___healthy",
    "Apple___Apple_scab",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Bacterial_spot",
    "Apple___Black_rot",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Peach___Bacterial_spot",
    "Apple___Cedar_apple_rust",
    "Tomato___Target_Spot",
    "Pepper,_bell___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Potato___Late_blight",
    "Tomato___Tomato_mosaic_virus",
    "Strawberry___healthy",
    "Apple___healthy",
    "Grape___Black_rot",
    "Potato___Early_blight",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Common_rust_",
    "Grape___Esca_(Black_Measles)",
    "Raspberry___healthy",
    "Tomato___Leaf_Mold",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Pepper,_bell___Bacterial_spot",
    "Corn_(maize)___healthy",
]
MODEL_INPUT_SIZE = 224
INPUT_DIM = (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)

THRESHOLD = 0.8

AREA = 500


def crop_and_resize(img, x, y, w, h):
    cropped_image = img[y : y + h, x : x + w]
    # resize the image to fit the model input shape
    resized_cropped_image = cv2.resize(
        cropped_image, INPUT_DIM, interpolation=cv2.INTER_AREA
    )
    resized_cropped_image = np.expand_dims(resized_cropped_image, axis=0)
    resized_cropped_image = resized_cropped_image.astype(np.float32)
    resized_cropped_image = resized_cropped_image / 255
    return resized_cropped_image


def tflite_predict(input_model, data):
    input_details = input_model.get_input_details()
    # print(input_details)
    output_details = input_model.get_output_details()
    input_model.set_tensor(0, data)
    input_model.invoke()
    output_data = input_model.get_tensor(output_details[0]["index"])
    return output_data


def detect_leaf(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # store the a-channel
    a_channel = lab[:, :, 1]
    # Automate threshold using Otsu method
    th = cv2.threshold(a_channel, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Mask the result with the original image
    masked = cv2.bitwise_and(img, img, mask=th)
    return masked, th


cap = cv2.VideoCapture(0)


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    imgContour = frame.copy()

    # imgContour = imgContour.astype(np.float32)
    # the masked image is the original image without the non green parts
    masked, mask = detect_leaf(imgContour)
    # find contours on the image
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    roi_list = []
    resized_cropped_images = []
    for c in contours:
        roi_dict = {}
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area > AREA:
            resized_cropped_image = crop_and_resize(masked, x, y, w, h)
            # resized_cropped_images.append(resized_cropped_image)
            # roi_dict["x"] = x
            # roi_dict["y"] = y
            # roi_dict["w"] = w
            # roi_dict["h"] = h
            # roi_list.append(roi_dict)
            preds = tflite_predict(tflite_model, resized_cropped_image)
            predicted_value = preds[0][np.argmax(preds[0])]
            if predicted_value > THRESHOLD:
                cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # preds = tflite_predict(tflite_model, resized_cropped_images)
    # for roi, pred in zip(roi_list, preds):
    #     pass
    cv2.imshow("Detection", imgContour)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break
