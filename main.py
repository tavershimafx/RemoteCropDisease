from qt_material import apply_stylesheet
import tensorflow as tf
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from keras.models import Model
import os
import sys
import time
from random import randint
import cv2
from PySide6 import QtCore
from PySide6.QtCore import Qt, QThread, Signal, Slot, QSize, QTimer
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap, QIcon, QColor
from PySide6 import QtWidgets, QtSvg, QtGui
from widgets import CustomDialog
import pandas as pd
from pathlib import Path
from datetime import datetime
from constants import CLASSES

from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QSlider,
    QProgressDialog,
    QListView,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
)
from detection_utils import *
from viewmodels.aircraft import Aircraft
import pygame

# qapp = QtWidgets.qApp

# Speed of the drone
S = 60


# load the model
tflite_model = tf.lite.Interpreter(model_path="resources\cropdisease.tflite")

# tflite_model.resize_tensor_input(0, [-1, 224, 224, 3])
tflite_model.allocate_tensors()

TFLITE_MODEL_PATH = "resources/cropdisease.tflite"
MODEL_INPUT_SIZE = 224
INPUT_DIM = (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)


def efficient_lite(img, detection_threshold):
    # Load the TFLite model
    options = ObjectDetectorOptions(
        num_threads=1,
        score_threshold=detection_threshold,
    )
    detector = ObjectDetector(model_path=TFLITE_MODEL_PATH, options=options)

    # Run object detection estimation using the model.
    detections = detector.detect(img)

    # Draw keypoints and edges on input image
    # image_np, predictions = visualize(img, detections)

    # Draw keypoints and edges on input image using classnames predicted by mobilent
    image_np, predictions = visualize_classnames_with_mobilenet(img, detections)
    return image_np, predictions


# def crop_and_resize(img, x, y, w, h):
#     """
#     This method crops the image at the cordinates given and resizes it to 224,224 which is the dimension the plant disease ai model accepts
#     Args:
#         img (_nparray_): image array
#         x (_int_): x cordinate of the top corner
#         y (_int_): y cordingate of the top corner
#         w (_int_): width of the image
#         h (_int_): height of the image

#     Returns:
#         _nparray_: resized image array
#     """
#     cropped_image = img[y : y + h, x : x + w]

#     # resize the image to fit the model input shape
#     resized_cropped_image = cv2.resize(
#         cropped_image, INPUT_DIM, interpolation=cv2.INTER_AREA
#     )
#     resized_cropped_image = np.expand_dims(resized_cropped_image, axis=0)
#     resized_cropped_image = resized_cropped_image.astype(np.float32)
#     resized_cropped_image = resized_cropped_image / 255
#     return resized_cropped_image


# def tflite_predict(input_model, data):
#     """

#     Args:
#         input_model (_kerasmodel_): _this is the loaded input model_
#         data (_nparray_): _this is the input image of shape 1,224,244,3_

#     Returns:
#         _nparray_: _this is the prediction of the network of shape 1,38 which contains the probability for all the 38 classes _
#     """
#     input_details = input_model.get_input_details()
#     # print(input_details)
#     output_details = input_model.get_output_details()
#     input_model.set_tensor(0, data)
#     input_model.invoke()
#     output_data = input_model.get_tensor(output_details[0]["index"])
#     return output_data


# def detect_leaf(img):
#     """
#     This method detects all the leaves in the image by dropping all non green colors and creating a mask on the green objects
#     Args:
#         img (_nparray_): _raw image from the camera_

#     Returns:
#         _nparray_: _mask with non green pixels set to 0,(black) and green pixels set to 255 _
#         _nparray_: _image with non green pixel set to 0 and green pixel left the way the are(this image would be given to the plant diesase detection ai/neural network)_
#     """
#     lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#     # store the a-channel
#     a_channel = lab[:, :, 1]
#     # Automate threshold using Otsu method
#     th = cv2.threshold(a_channel, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#     # Mask the result with the original image
#     masked = cv2.bitwise_and(img, img, mask=th)
#     return masked, th


class Thread(QThread):
    updateFrame = Signal(QImage)
    prediction_dict = Signal(dict)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.trained_file = None
        self.status = True

        self.isPredict = False
        self.minArea = 500
        # if the model is not minProbability sure about a prediction it shouldn't return
        # or draw the prediction
        self.minProbability = 0.2
        # list of the predcitions gotten from the frames
        self.predictions = []

        self.aircraft = Aircraft()

        #self.no_connection_image = no_connection_image.toImage()

    def set_minArea(self, area):
        self.minArea = area

    def set_minProbability(self, probability):
        self.minProbability = float(probability)

    def start_stop_predictions(self):
        self.isPredict = not self.isPredict

    def run(self):
        """
        This function extracts the frames one by one
        from the self.cap attribute and process them
        detecting and drawing bounding boxes on the frames
        and emiting each frame to the MainWindow
        """
        while self.status:
            # substitute this for the drone camera feed
            # ret, frame = self.cap.read()
            frame_read = self.aircraft.get_frame()  # get a frame from the aircraft
            if frame_read == None:
                # self.start_stop_predictions()
                # self.updateFrame.emit(scaled_img)
                continue

            frame = frame_read.frame

            # this happens when we lost the video feed from the drone
            if type(frame) is not ndarray:
                continue

            # copy the frame to avoid making changes to the orignal frames
            imgContour = frame.copy()
            img_detections, predictions = efficient_lite(frame, self.minProbability)
            for prediction in predictions:
                self.prediction_dict.emit(prediction)
            # the masked image is the original image without the non green parts
            # masked, mask = detect_leaf(imgContour)
            # # find contours on the image
            # ret, thresh = cv2.threshold(mask, 127, 255, 0)
            # contours, hierarchy = cv2.findContours(thresh, 1, 2)
            # if self.isPredict:
            #     for c in contours:
            #         prediction_dict = {}
            #         roi_dict = {}
            #         peri = cv2.arcLength(c, True)
            #         approx = cv2.approxPolyDP(c, 0.01 * peri, True)
            #         x, y, w, h = cv2.boundingRect(c)
            #         area = cv2.contourArea(c)
            #         if area > self.minArea:
            #             resized_cropped_image = crop_and_resize(masked, x, y, w, h)
            #             preds = tflite_predict(tflite_model, resized_cropped_image)

            #             predicted_value = preds[0][np.argmax(preds[0])]
            #             leaf_type = CLASSES[np.argmax(preds[0])]
            #             prediction_dict["leaf"] = leaf_type
            #             prediction_dict["probability"] = predicted_value
            #             self.predictions.extend(prediction_dict)
            #             if predicted_value > self.minProbability:
            #                 prob = round(predicted_value, 2)
            #                 text = f"{leaf_type} {prob}"
            #                 # draw the rectangles on the frame
            #                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #                 cv2.putText(
            #                     frame,
            #                     text,
            #                     (x + (w // 2) - 10, y + (h // 2) - 10),
            #                     cv2.FONT_HERSHEY_COMPLEX,
            #                     0.7,
            #                     (0, 255, 0),
            #                     1,
            #                 )
            #                 self.prediction_dict.emit(prediction_dict)
            # Reading the image in RGB to display it
            # 🥸 AKO JOGODO abeg help me comment this line, try the next one make i see wetin go happen
            color_frame = cv2.cvtColor(img_detections, cv2.COLOR_BGR2RGB)
            # color_frame = img_detections
            # Creating and scaling QImage
            h, w, ch = color_frame.shape
            img = QImage(color_frame.data, w, h, ch * w, QImage.Format_RGB888)
            scaled_img = img.scaled(1200, 600, Qt.KeepAspectRatio)

            # Emit signal
            self.updateFrame.emit(scaled_img)
        sys.exit(-1)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Title and dimensions
        self.setWindowTitle("Leaf Disease Detection")
        self.setGeometry(0, 0, 800, 700)

        # Drone control variables
        self.isStart = False
        self.isPredict = False

        self.minArea = 500
        self.minProbability = 0.6

        # holds the names of all the leafs predicted to be exported to csv
        self.leafs = []
        # holds the probabilities of the predictions to be exported to csv
        self.probabilities = []

        # initialize a default. Its uncertain if any joystick is available so we assign None
        self.joystick = None
        pygame.init()  # Init pygame to enable joystick and media objects

        # Main menu bar
        self.menu = self.menuBar()
        self.menu_file = self.menu.addMenu("File")
        self.pixmap = QtGui.QPixmap("resources/images/nowifi.jpeg")
        exit = QAction("Exit", self, triggered=app.quit)
        self.menu_file.addAction(exit)

        self.menu_about = self.menu.addMenu("&About")
        about = QAction(
            "About Qt",
            self,
            shortcut=QKeySequence(QKeySequence.HelpContents),
            triggered=app.aboutQt,
        )
        self.menu_about.addAction(about)

        # Create a label for the display camera
        self.label = QLabel(self)
        self.label.setFixedSize(1070, 600)

        # Thread in charge of updating the image
        self.th = Thread(self)
        self.th.finished.connect(self.close)
        self.th.updateFrame.connect(self.setImage)
        self.th.prediction_dict.connect(self.updatePredictionList)
        # Model group
        self.group_model = QGroupBox("Prediction parameters")
        # self.group_model.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.group_model.setFixedWidth(1070)
        model_layout = QVBoxLayout()

        top_slider_layout = QHBoxLayout()
        bottom_slider_layout = QHBoxLayout()
        # this is the slider that controls the area value
        self.areaSlider = QSlider(orientation=Qt.Orientation.Horizontal)
        # this is the threshold slider that controls the minimum probability to be considered by the network
        self.thresholdSlider = QSlider(orientation=Qt.Orientation.Horizontal)
        self.areaSlider.setMinimum(100)
        self.areaSlider.setMaximum(1000)
        self.areaSlider.setTickInterval(50)
        self.thresholdSlider.setMinimum(0)
        self.thresholdSlider.setMaximum(100)
        self.thresholdSlider.setTickInterval(10)

        top_slider_layout.addWidget(QLabel("Area"), 5)
        self.areaLabel = QLabel(f"{self.minArea}")
        self.areaMinLabel = QLabel("         100")
        self.areaMaxLabel = QLabel("1000")
        top_slider_layout.addWidget(self.areaLabel, 2)
        top_slider_layout.addWidget(self.areaMinLabel, 5)
        top_slider_layout.addWidget(self.areaSlider, 83)
        top_slider_layout.addWidget(self.areaMaxLabel, 5)

        self.thresholdLabel = QLabel(f"{self.minProbability}")
        self.thresholdMinLabel = QLabel("         0")
        self.thresholdMaxLabel = QLabel("1")

        self.areaSlider.setValue(self.minArea)
        self.thresholdSlider.setValue(self.minProbability)
        bottom_slider_layout.addWidget(QLabel("Threshold"), 2)
        bottom_slider_layout.addWidget(self.thresholdLabel, 2)
        bottom_slider_layout.addWidget(self.thresholdMinLabel, 2)
        bottom_slider_layout.addWidget(self.thresholdSlider, 88)
        bottom_slider_layout.addWidget(self.thresholdMaxLabel, 5)

        model_layout.addLayout(top_slider_layout, 1)
        model_layout.addLayout(bottom_slider_layout, 1)
        self.group_model.setLayout(model_layout)

        # Buttons layout
        buttons_layout = QHBoxLayout()
        self.connect_drone_button = QPushButton("")
        self.connect_pad_button = QPushButton("")
        self.connect_drone_button.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Expanding
        )

        # Load the drone svg
        drone_svg_renderer = QtSvg.QSvgRenderer(
            "resources/images/drone-svgrepo-com-black.svg"
        )
        # Prepare a QImage with desired characteritisc
        self.drone_svg = QtGui.QImage(500, 500, QtGui.QImage.Format_ARGB32)
        # Get QPainter that paints to the image
        drone_svg_painter = QtGui.QPainter(self.drone_svg)
        drone_svg_renderer.render(drone_svg_painter)

        # drone color
        self.drone_icon_color_green = QColor(0, 255, 0)
        self.drone_icon_color_gray = QColor(211, 211, 211)
        self.drone_icon_green = self.drone_icon_colored(self.drone_icon_color_green)
        self.drone_icon_gray = self.drone_icon_colored(self.drone_icon_color_gray)
        self.connect_drone_button.setIcon(self.drone_icon_gray)
        self.connect_drone_button.setIconSize(QSize(30, 30))

        # load the pad svg
        pad_svg_renderer = QtSvg.QSvgRenderer(
            "resources/images/gamepad-with-wire-svgrepo-com.svg"
        )
        # Prepare a QImage with desired characteritisc
        self.pad_svg = QtGui.QImage(500, 500, QtGui.QImage.Format_ARGB32)
        # Get QPainter that paints to the image
        pad_svg_painter = QtGui.QPainter(self.pad_svg)
        pad_svg_renderer.render(pad_svg_painter)
        # drone color
        self.pad_icon_color_green = QColor(0, 255, 0)
        self.pad_icon_color_gray = QColor(211, 211, 211)
        self.pad_icon_green = self.pad_icon_colored(self.pad_icon_color_green)
        self.pad_icon_gray = self.pad_icon_colored(self.pad_icon_color_gray)
        self.connect_pad_button.setIcon(self.pad_icon_gray)
        self.connect_pad_button.setIconSize(QSize(30, 30))
        self.connect_pad_button.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Expanding
        )

        self.connect_pad_button.setFixedHeight(100)
        self.connect_drone_button.setFixedHeight(100)
        buttons_layout.addWidget(self.connect_drone_button)
        buttons_layout.addWidget(self.connect_pad_button)

        bottom_layout = QVBoxLayout()
        bottom_layout.addWidget(self.group_model, 1)

        # Right layout
        right_layout = QVBoxLayout()

        right_layout.addWidget(self.label)
        right_layout.addLayout(bottom_layout)

        # box responsible for displaying the velocity of the drone
        self.velocity_altitude_group = QGroupBox("Drone Stats")
        velocity_altitude_group_layout = QVBoxLayout()
        self.veloctiy_label = QLabel("Velocity  0cm/s")
        self.altitude_label = QLabel("Altitude  0m")
        self.battery_percentage = QLabel("Battery 58%")
        velocity_altitude_group_layout.addWidget(self.veloctiy_label)
        velocity_altitude_group_layout.addWidget(self.altitude_label)
        velocity_altitude_group_layout.addWidget(self.battery_percentage)
        self.velocity_altitude_group.setLayout(velocity_altitude_group_layout)
        self.velocity_altitude_group.setFixedHeight(150)

        self.predictions_group = QGroupBox("Top Predictions")
        self.export_to_csv_button = QPushButton("Export to CSV")
        self.csv_btn_layout = QHBoxLayout()
        self.csv_btn_layout.addWidget(self.export_to_csv_button)
        self.csv_btn_layout.addStretch()
        self.predictions_group_layout = QVBoxLayout()
        self.predictions_group_layout.addLayout(self.csv_btn_layout)
        self.scroll = QScrollArea(self)
        self.scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.predictions_layout = QVBoxLayout()
        self.predictions_group_layout.addStretch()
        self.predictions_group.setLayout(self.predictions_group_layout)
        self.predictions_group.setFixedHeight(410)

        start_predict_button_layout = QHBoxLayout()
        self.start_button_text = "ON"
        self.start_stop_button = QPushButton(self.start_button_text)
        self.predict_button = QPushButton("PREDICT")
        start_predict_button_layout.addWidget(self.start_stop_button)
        start_predict_button_layout.addWidget(self.predict_button)

        left_layout = QVBoxLayout()
        left_layout.addLayout(buttons_layout)
        left_layout.addWidget(self.velocity_altitude_group, 1)
        # left_layout.addWidget(self.altitude_group, 1)
        left_layout.addWidget(self.predictions_group, 1)
        left_layout.addLayout(start_predict_button_layout)

        # Main Layout
        layout = QHBoxLayout()
        # add the right layout
        layout.addLayout(right_layout)
        layout.addSpacing(30)
        # add the left layout
        layout.addLayout(left_layout)

        # Central widget
        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # no need for manual connection again
        self.connect_drone_button.clicked.connect(self.connect_drone)
        self.predict_button.clicked.connect(self.predict_start_stop)
        self.export_to_csv_button.clicked.connect(self.export_to_csv)
        self.thresholdSlider.valueChanged.connect(self.thresholdChange)
        self.areaSlider.valueChanged.connect(self.areaChange)
        self.setNoWifi()

        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.joystick_handler)
        self.timer.start(20)

        self.start()

    @Slot()
    def set_model(self, text):
        self.th.set_file(text)

    def joystick_handler(self):
        battery = "Battery: {}%".format(self.th.aircraft.get_battery())
        altitude = "Altitude: {}cm".format(self.th.aircraft.get_altitude())
        self.battery_percentage.setText(battery)
        self.altitude_label.setText(altitude)

        for event in pygame.event.get():
            if event.type == pygame.USEREVENT + 1:
                self.th.aircraft.update()
                #
            elif event.type == pygame.JOYBUTTONDOWN:
                self.keyPressed(event.button)
            elif event.type == pygame.JOYBUTTONUP:
                self.keyReleased(event.button)
            elif event.type == pygame.JOYDEVICEADDED:
                # This event will be generated when the program starts for any
                # joystick, automatically detecting it without needing to create it manually.
                self.joystick = pygame.joystick.Joystick(event.device_index)
                self.connect_pad_button.setIcon(
                    self.pad_icon_green
                )  # make the pad icon green to signify active
            elif event.type == pygame.JOYDEVICEREMOVED:
                self.joystick = None
                self.connect_pad_button.setIcon(
                    self.pad_icon_gray
                )  # make the pad icon grey to signify inactive

    def keyPressed(self, key):
        """Update velocities based on key pressed
        Arguments:
            key：an integer value identifying the joystick keyId or axis keyId that was pressed.
        """
        # attempt takeoff is not already
        if not self.th.aircraft.initite_takeoff():
            return

        if key == 0:  # Triangle key
            self.th.aircraft.move(forward_back=S)  # set forward velocity
        elif key == 1:  # Circle key
            self.th.aircraft.move(left_right=S)  # set right velocity
        elif key == 2:  # Times key
            self.th.aircraft.move(forward_back=-S)  # set backward velocity
        elif key == 3:  # Square key
            self.th.aircraft.move(left_right=-S)  # set left velocity
        elif key == 4:  # Left 1
            self.th.aircraft.move(up_down=S)  # set up velocity
        elif key == 5:  # Right 1 key
            self.th.aircraft.move(yaw=S)  # set yaw right velocity
        elif key == 6:  # Left 2 key
            self.th.aircraft.move(up_down=-S)  # set down velocity
        elif key == 7:  # Right 2 key
            self.th.aircraft.move(yaw=-S)  # set yaw left velocity
        elif key == 10:  # left steer button
            self.th.aircraft.stream_video()  # start streming video
        elif key == 11:  # Right steer button
            self.th.aircraft.capture_image()  # take a snapshot

    def keyReleased(self, key):
        """Update velocities based on key pressed
        Arguments:
            key：an integer value identifying the joystick keyId or axis keyId that was pressed.
        """

        if key == 0:  # Triangle key
            self.th.aircraft.move()  # set forward velocity
        elif key == 1:  # Circle key
            self.th.aircraft.move()  # set right velocity
        elif key == 2:  # Times key
            self.th.aircraft.move()  # set backward velocity
        elif key == 3:  # Square key
            self.th.aircraft.move()  # set left velocity
        elif key == 4:  # Left 1
            self.th.aircraft.move()  # set up velocity
        elif key == 5:  # Right 1 key
            self.th.aircraft.move()  # set yaw right velocity
        elif key == 6:  # Left 2 key
            self.th.aircraft.move()  # set down velocity
        elif key == 7:  # Right 2 key
            self.th.aircraft.move()  # set yaw left velocity
        elif key == 8:  # select key
            self.th.aircraft.initite_land()
        elif key == 9:  # start key
            self.th.aircraft.initite_takeoff()

    def try_init_aircraft_on_error(self, response):
        """Try to reinitialize the aircraft object if it failed to respond to command.
        This usually happens when the aircraft crashes physically but remains on.
        """
        if response == "error":
            self.th.aircraft = Aircraft()

    @Slot()
    def kill_thread(self):
        print("Finishing...")
        self.th.cap.release()
        cv2.destroyAllWindows()
        self.status = False
        self.th.terminate()

        # Give time for the thread to finish
        time.sleep(1)

    @Slot()
    def start(self):
        print("Starting...")
        self.th.start()

    @Slot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    @Slot(dict)
    def updatePredictionList(self, prediction_dict):
        """

        Args:
            prediction_dict (_type_): _description_
        This method connects to the predictions signals emmitted by the thread
        and updates the Predictions group box with all the predictions by the network
        """
        print(f"This is the prediction dictionary {prediction_dict}")
        leaf = prediction_dict["leaf"]
        probability = prediction_dict["probability"]
        probability = round(probability, 2)
        label_layout = QHBoxLayout()
        label_leaf = QLabel(f"{leaf}")
        label_probability = QLabel(f"{probability}")
        label_probability.setStyleSheet("color:green")
        self.leafs.append(leaf)
        self.probabilities.append(probability)
        label_layout.addWidget(label_leaf)
        label_layout.addStretch()
        label_layout.addWidget(label_probability)
        self.predictions_layout.addLayout(label_layout)
        widget = QWidget()
        widget.setLayout(self.predictions_layout)
        self.scroll.setWidget(widget)
        # scroll.setWidgetResizable(True)
        # scroll.setFixedHeight(300)
        self.predictions_group_layout.addWidget(self.scroll)
        self.predictions_group_layout.addStretch()

    @Slot()
    def export_to_csv(self):
        """
        This method exports the
        """
        if len(self.probabilities) != 0 and len(self.leafs) != 0:
            # data
            data = {"leaf": self.leafs, "probability": self.probabilities}
            df = pd.DataFrame(data)
            # desktop path on unix
            # desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
            # desktop path on windows
            # desktop = os.path.join(os.path.join(os.environ["USERPROFILE"]), "Desktop")
            desktop = os.path.expanduser("~/Desktop")
            # get the current date time
            current_dateTime = datetime.now()
            filepath = Path(
                f"{desktop}/leaf Disease Detection/predictions/result{current_dateTime}.csv"
            )
            filepath.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(filepath)

    @Slot()
    def areaChange(self):
        """
        This method handles the changes to the areavalue when the area slider is moved
        """
        # area is a value between 500 and 1000
        # get the value
        self.minArea = self.areaSlider.value()
        # set the value on the label text
        self.areaLabel.setText(f"{self.minArea}")
        # set change the area value on the predicton thread
        self.th.set_minArea(self.minArea)

    @Slot()
    def thresholdChange(self):
        """
        This method handles changes to the minimum porbability value when the threshold slider is moved
        """
        # threshold is a value between 1 and 100
        # get the value from the threshold slider
        threshold = self.thresholdSlider.value()
        # scale the value by dividing it by 100
        threshold = threshold / 100
        self.minProbability = threshold
        # set the value on the threshold label text
        self.thresholdLabel.setText(f"{self.minProbability }")
        self.th.set_minProbability(f"{self.minProbability }")

    def setNoWifi(self):
        """
        This method sets The wifi icon when the drone is not connected
        """
        # pixmap = QtGui.QPixmap("resources/images/nowifi.jpeg")
        self.label.setPixmap(self.pixmap)
        # self.setStyleSheet("text-align:center")
        self.label.setStyleSheet(f"qproperty-alignment: {int(QtCore.Qt.AlignCenter)};")

    @Slot()
    def connect_drone(self):
        """
        This method handles connecting the drone
        if the connection is successful exec the drone_connection_success dialog
        and set the drone_button_icon to drone_icon_green
        else
        show the drone_connection_failed dialog and set the drone_button_icon to drone_icon_gray (optional)
        """
        self.drone_connection_success = CustomDialog(
            title="Success", content="drone connection success"
        )
        self.drone_connection_failed = CustomDialog(
            title="Error", content="drone connection failed"
        )

        # try connecting to the pad
        # if it succeeeds show
        try:
            self.th.aircraft.connect()
            self.connect_drone_button.setIcon(self.drone_icon_green)
            self.start_stop_button.setText("CONNECTED")
            self.drone_connection_success.exec()
        except:
            self.start_stop_button.setText("DISCONNECTED")
            self.drone_connection_failed.exec()

    @Slot()
    def predict_start_stop(self):
        """
        This method starts or stops the predictions
        """
        self.drone_not_connected = CustomDialog(
            title="Error",
            content="Drone is not connected. Please make sure you connect to the drone's wifi",
        )
        if self.th.aircraft.is_connected:  # .isDroneConnected:
            # start predictions
            if self.isPredict:
                # stop predictions
                self.isPredict = False
                self.th.start_stop_predictions()
                # set start stop button text to on
                self.predict_button.setText("PREDICT")
            else:
                # start the drone
                self.isPredict = True
                self.th.start_stop_predictions()
                # set start stop button text to off
                self.predict_button.setText("STOP")
        else:
            self.drone_not_connected.exec()
            self.drone_icon_color = QColor(255, 0, 0)

    # method to set the colored version of the connect drone icon
    def drone_icon_colored(self, color):

        # Copy the image
        new_image = self.drone_svg.copy()

        # We are going to paint a plain color over the alpha
        paint = QtGui.QPainter()
        paint.begin(new_image)
        paint.setCompositionMode(QtGui.QPainter.CompositionMode_SourceIn)
        paint.fillRect(new_image.rect(), color)
        paint.end()

        return QIcon(QPixmap.fromImage(new_image))

    # method to set the colored version of the connect pad icon
    def pad_icon_colored(self, color):

        # Copy the image
        new_image = self.pad_svg.copy()

        # We are going to paint a plain color over the alpha
        paint = QtGui.QPainter()
        paint.begin(new_image)
        paint.setCompositionMode(QtGui.QPainter.CompositionMode_SourceIn)
        paint.fillRect(new_image.rect(), color)
        paint.end()

        return QIcon(QPixmap.fromImage(new_image))


if __name__ == "__main__":
    # sys.argv += ["-platform", "windows:darkmode=2"]
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="dark_teal.xml")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
