from qt_material import apply_stylesheet
import tensorflow as tf
import numpy as np
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
from constants import CLASSES

# from src.UI import ui_main_window
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

from viewmodels.aircraft import Aircraft
import pygame
# qapp = QtWidgets.qApp

# Speed of the drone
S = 60

"""This example uses the video from a  webcam to apply pattern
detection from the OpenCV module. e.g.: face, eyes, body, etc."""
# load the model
tflite_model = tf.lite.Interpreter(model_path="resources/plant_diseas_model.tflite")

# tflite_model.resize_tensor_input(0, [-1, 224, 224, 3])
tflite_model.allocate_tensors()


MODEL_INPUT_SIZE = 224
INPUT_DIM = (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)


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


class Thread(QThread):
    updateFrame = Signal(QImage)
    prediction_dict = Signal(dict)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.trained_file = None
        self.status = True
        #self.cap = True
        self.isPredict = False
        self.minArea = 500
        # if the model is not minProbability sure about a prediction it shouldn't return
        # or draw the prediction
        self.minProbability = 0.8
        # list of the predcitions gotten from the frames
        self.predictions = []

        # initialize the aircraft which we will be communicating to
        self.aircraft = Aircraft()

    def set_file(self, fname):
        # The data comes with the 'opencv-python' module
        self.trained_file = os.path.join(cv2.data.haarcascades, fname)

    def set_minArea(self, area):
        self.minArea = area

    def set_minProbability(self, probability):
        self.minProbability = probability

    def start_stop_predictions(self):
        self.isPredict = not self.isPredict

    def run(self):
        #self.cap = cv2.VideoCapture(0)
        while self.status:
            # substitute this for the drone camera feed
            # ret, frame = self.cap.read()
            frame_read = self.aircraft.get_frame() # get a frame from the aircraft
            if frame_read == None:
                continue

            frame = frame_read.frame
            # copy the frame to avoid making changes to the orignal frames
            imgContour = frame.copy()
            # the masked image is the original image without the non green parts
            masked, mask = detect_leaf(imgContour)
            # find contours on the image
            ret, thresh = cv2.threshold(mask, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, 1, 2)
            roi_list = []
            resized_cropped_images = []
            if self.isPredict:
                for c in contours:
                    prediction_dict = {}
                    roi_dict = {}
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.01 * peri, True)
                    x, y, w, h = cv2.boundingRect(c)
                    area = cv2.contourArea(c)
                    if area > self.minArea:
                        resized_cropped_image = crop_and_resize(masked, x, y, w, h)
                        # resized_cropped_images.append(resized_cropped_image)
                        # roi_dict["x"] = x
                        # roi_dict["y"] = y
                        # roi_dict["w"] = w
                        # roi_dict["h"] = h
                        # roi_list.append(roi_dict)
                        preds = tflite_predict(tflite_model, resized_cropped_image)

                        predicted_value = preds[0][np.argmax(preds[0])]
                        leaf_type = CLASSES[np.argmax(preds[0])]
                        prediction_dict["leaf"] = leaf_type
                        prediction_dict["probability"] = predicted_value
                        self.predictions.extend(prediction_dict)
                        if predicted_value > self.minProbability:
                            # draw the rectangles on the frame
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            self.prediction_dict.emit(prediction_dict)
            # Reading the image in RGB to display it
            color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
        # self.setGeometry(0, 0, 800, 500)
        self.setGeometry(0, 0, 800, 700)
        # Drone control variables
        self.isStart = False
        self.isPredict = False
        #self.isDroneConnected = True
        #self.isPadConnected = False
        #self.isDroneOn = False

        
        
        # initialize a default. Its uncertain if any joystick is available so we assign None
        self.joystick = None
        pygame.init() # Init pygame to enable joystick and media objects

        # Main menu bar
        self.menu = self.menuBar()
        self.menu_file = self.menu.addMenu("File")
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

        # set default values for area and threshold
        self.area = 0
        self.threshold = 0

        self.areaSlider = QSlider(orientation=Qt.Orientation.Horizontal)
        self.thresholdSlider = QSlider(orientation=Qt.Orientation.Horizontal)
        self.areaSlider.setMinimum(0)
        self.areaSlider.setMaximum(100)
        self.areaSlider.setTickInterval(1)
        self.thresholdSlider.setMinimum(500)
        self.thresholdSlider.setMaximum(1000)
        self.thresholdSlider.setTickInterval(50)

        # for xml_file in os.listdir(cv2.data.haarcascades):
        #     if xml_file.endswith(".xml"):
        #         self.combobox.addItem(xml_file)

        top_slider_layout.addWidget(QLabel("Area"), 5)
        self.areaLabel = QLabel("0")
        self.areaMinLabel = QLabel("         0")
        self.areaMaxLabel = QLabel("1")
        top_slider_layout.addWidget(self.areaLabel, 2)
        top_slider_layout.addWidget(self.areaMinLabel, 5)
        top_slider_layout.addWidget(self.areaSlider, 83)
        top_slider_layout.addWidget(self.areaMaxLabel, 5)

        self.thresholdLabel = QLabel("1000")
        self.thresholdMinLabel = QLabel("         500")
        self.thresholdMaxLabel = QLabel("1000")
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
        # self.drone_svg = QIcon(QIcon("resources/images/drone-svgrepo-com-black.svg"))

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
        # right_layout.addLayout(buttons_layout, 1)

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

        # box responsible for displaying the altitude of the drone
        # self.altitude_group = QGroupBox("Altitude")
        # altitude_group_layout = QVBoxLayout()
        # self.altitude_label = QLabel("0cm")
        # altitude_group_layout.addWidget(self.altitude_label)
        # self.altitude_group.setLayout(altitude_group_layout)
        # self.altitude_group.setFixedHeight(150)

        self.predictions_group = QGroupBox("Top Predictions")
        self.export_to_csv_button = QPushButton("Export to CSV")
        self.csv_btn_layout = QHBoxLayout()
        self.csv_btn_layout.addWidget(self.export_to_csv_button)
        self.csv_btn_layout.addStretch()
        self.predictions_group_layout = QVBoxLayout()
        self.predictions_group_layout.addLayout(self.csv_btn_layout)
        # predictions = [
        #     {"leaf": "apple", "probability": "90%"},
        #     {"leaf": "corn", "probability": "70%"},
        #     {"leaf": "tomato", "probability": "80%"},
        #     {"leaf": "soyabeans healthy", "probability": "90%"},
        #     {"leaf": "strawberry leaf scorch", "probability": "60%"},
        #     {"leaf": "peach healthy", "probability": "80%"},
        #     {"leaf": "Blue berry", "probability": "99%"},
        #     {"leaf": "Raspberry healthy", "probability": "88%"},
        #     {"leaf": "cherry healthy", "probability": "96%"},
        #     {"leaf": "cherry healthy", "probability": "96%"},
        #     {"leaf": "cherry healthy", "probability": "96%"},
        #     {"leaf": "cherry healthy", "probability": "96%"},
        #     {"leaf": "cherry healthy", "probability": "96%"},
        #     {"leaf": "cherry healthy", "probability": "96%"},
        #     {"leaf": "cherry healthy", "probability": "96%"},
        #     {"leaf": "cherry healthy", "probability": "96%"},
        # ]
        self.scroll = QScrollArea(self)
        self.scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.predictions_layout = QVBoxLayout()
        # for prediction_dict in predictions:
        #     leaf = prediction_dict["leaf"]
        #     probability = prediction_dict["probability"]
        #     label_layout = QHBoxLayout()
        #     label_leaf = QLabel(f"{leaf}")
        #     label_probability = QLabel(f"{probability}")
        #     label_probability.setStyleSheet("color:green")
        #     label_layout.addWidget(label_leaf)
        #     label_layout.addStretch()
        #     label_layout.addWidget(label_probability)
        #     self.predictions_layout.addLayout(label_layout)
        # widget = QWidget()
        # widget.setLayout(self.predictions_layout)
        # self.scroll.setWidget(widget)
        # # scroll.setWidgetResizable(True)
        # # scroll.setFixedHeight(300)
        # self.predictions_group_layout.addWidget(self.scroll)
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
        # self.start()

        #self.connect_pad_button.clicked.connect(self.connect_pad) # the pad is detected automatically now
                                                                    #no need for manual connection again
        self.connect_drone_button.clicked.connect(self.connect_drone)
        #self.start_stop_button.clicked.connect(self.start_stop_drone)
        self.predict_button.clicked.connect(self.predict_start_stop)
        self.setNoWifi()
        self.start()

        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.joystick_handler)
        self.timer.start(20)
        # Connections
        # self.button1.clicked.connect(self.start)
        # self.button2.clicked.connect(self.kill_thread)
        # self.connect_pad_button.setEnabled(False)
        # self.combobox.currentTextChanged.connect(self.set_model)
   
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
                response = self.th.aircraft.update()
                self.try_init_aircraft_on_error(response)
            elif event.type == pygame.JOYBUTTONDOWN:
                self.keyPressed(event.button)
            elif event.type == pygame.JOYBUTTONUP:
                self.keyReleased(event.button)
            elif event.type == pygame.JOYDEVICEADDED:
                # This event will be generated when the program starts for any
                # joystick, automatically detecting it without needing to create it manually.
                self.joystick = pygame.joystick.Joystick(event.device_index)
                self.connect_pad_button.setIcon(self.pad_icon_green) # make the pad icon green to signify active
            elif event.type == pygame.JOYDEVICEREMOVED:
                self.joystick = None
                self.connect_pad_button.setIcon(self.pad_icon_gray) # make the pad icon grey to signify inactive

    def keyPressed(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key：an integer value identifying the joystick keyId or axis keyId that was pressed.
        """
        # attempt takeoff is not already
        if not self.th.aircraft.initite_takeoff():
            return

        if key == 0: # Triangle key
            response = self.th.aircraft.move(forward_back=S)  # set forward velocity
            self.try_init_aircraft_on_error(response)
        elif key == 1: # Circle key
            response = self.th.aircraft.move(left_right=S)  # set right velocity
            self.try_init_aircraft_on_error(response)
        elif key == 2: # Times key
            response = self.th.aircraft.move(forward_back=-S) # set backward velocity
            self.try_init_aircraft_on_error(response)
        elif key == 3: # Square key
            response = self.th.aircraft.move(left_right=-S)  # set left velocity
            self.try_init_aircraft_on_error(response)
        elif key == 4: # Left 1
            response = self.th.aircraft.move(up_down=S)  # set up velocity
            self.try_init_aircraft_on_error(response)
        elif key == 5: # Right 1 key
            response = self.th.aircraft.move(yaw=S)  # set yaw right velocity
            self.try_init_aircraft_on_error(response)
        elif key == 6: # Left 2 key
            response = self.th.aircraft.move(up_down=-S)  # set down velocity
            self.try_init_aircraft_on_error(response)
        elif key == 7: # Right 2 key
            response = self.th.aircraft.move(yaw=-S)  # set yaw left velocity
            self.try_init_aircraft_on_error(response)
        elif key == 10: # left steer button
            response = self.th.aircraft.stream_video() # start streming video
            self.try_init_aircraft_on_error(response)
        elif key == 11: # Right steer button
            response = self.th.aircraft.capture_image() # take a snapshot
            self.try_init_aircraft_on_error(response)
        
    def keyReleased(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key：an integer value identifying the joystick keyId or axis keyId that was pressed.
        """
        
        if key == 0: # Triangle key
            response = self.th.aircraft.move()  # set forward velocity
            self.try_init_aircraft_on_error(response)
        elif key == 1: # Circle key
            response = self.th.aircraft.move()  # set right velocity
            self.try_init_aircraft_on_error(response)
        elif key == 2: # Times key
            response = self.th.aircraft.move() # set backward velocity
            self.try_init_aircraft_on_error(response)
        elif key == 3: # Square key
            response = self.th.aircraft.move()  # set left velocity
            self.try_init_aircraft_on_error(response)
        elif key == 4: # Left 1
            response = self.th.aircraft.move()  # set up velocity
            self.try_init_aircraft_on_error(response)
        elif key == 5: # Right 1 key
            response = self.th.aircraft.move()  # set yaw right velocity
            self.try_init_aircraft_on_error(response)
        elif key == 6: # Left 2 key
            response = self.th.aircraft.move()  # set down velocity
            self.try_init_aircraft_on_error(response)
        elif key == 7: # Right 2 key
            response = self.th.aircraft.move()  # set yaw left velocity
            self.try_init_aircraft_on_error(response)
        elif key == 8:  # select key
            response = self.th.aircraft.initite_land()
            self.try_init_aircraft_on_error(response)
        elif key == 9:  # start key
            response = self.th.aircraft.initite_takeoff()
            self.try_init_aircraft_on_error(response)
     
    def try_init_aircraft_on_error(self, response):
        """ Try to reinitialize the aircraft object if it failed to respond to command.
            This usually happens when the aircraft crashes physically but remains on.
        """
        if response == 'error':
            self.th.aircraft = Aircraft()
    
    @Slot()
    def kill_thread(self):
        print("Finishing...")
        # self.button2.setEnabled(False)
        # self.button1.setEnabled(True)
        self.th.cap.release()
        cv2.destroyAllWindows()
        self.status = False
        self.th.terminate()
        # Give time for the thread to finish
        time.sleep(1)

    @Slot()
    def start(self):
        print("Starting...")
        # self.button2.setEnabled(True)
        # self.button1.setEnabled(False)
        # self.th.set_file(self.combobox.currentText())
        self.th.start()

    @Slot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    @Slot(dict)
    def updatePredictionList(self, prediction_dict):
        print(f"This is the prediction dictionary {prediction_dict}")
        leaf = prediction_dict["leaf"]
        probability = prediction_dict["probability"]
        label_layout = QHBoxLayout()
        label_leaf = QLabel(f"{leaf}")
        label_probability = QLabel(f"{probability}")
        label_probability.setStyleSheet("color:green")
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

    def setNoWifi(self):
        pixmap = QtGui.QPixmap("resources/images/nowifi.jpeg")
        self.label.setPixmap(pixmap)
        # self.setStyleSheet("text-align:center")
        self.label.setStyleSheet(f"qproperty-alignment: {int(QtCore.Qt.AlignCenter)};")

    @Slot()
    def change_drone_icon_color(self):

        pass

    # @Slot()
    # def connect_pad(self):
    #     self.pad_connection_success = CustomDialog(
    #         title="Pad connection", content="Pad connection success"
    #     )
    #     self.pad_connection_failed = CustomDialog(
    #         title="Pad connection", content="Pad connection failed"
    #     )
    #     # try connecting to the pad
    #     # if it succeeeds show
    #     # self.pad_connection_success.exec()
    #     # self.connect_pad_button.setIcon(self.pad_icon_gray)
    #     self.connect_pad_button.setIcon(self.pad_icon_green)
    #     # else show
    #     self.pad_connection_success.exec()

    @Slot()
    def connect_drone(self):
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


    # @Slot()
    # def start_stop_drone(self):
    #     self.drone_not_connected = CustomDialog(
    #         title="Error",
    #         content="Drone is not connected. Please make sure you connect to the drone's wifi",
    #     )
    #     if not self.th.aircraft.is_connected: #.isDroneConnected:
    #         self.th.aircraft.connect()
    #         self.start_stop_button.setText("CONNECTED")
    #         #start up drone
    #         if self.isDroneOn:
    #             # stop the drone
    #             self.isDroneOn = False
    #             # set start stop button text to on
    #             self.start_stop_button.setText("ON")
    #         else:
    #             # start the drone
    #             self.isDroneOn = True
    #             # set start stop button text to off
    #             self.start_stop_button.setText("OFF")
    #         print(self.start_stop_button.text())
    #     else:
    #         self.drone_not_connected.exec()
    #         self.start_stop_button.setText("DISCONNECTED")

    @Slot()
    def predict_start_stop(self):
        self.drone_not_connected = CustomDialog(
            title="Error",
            content="Drone is not connected. Please make sure you connect to the drone's wifi",
        )
        if self.th.aircraft.is_connected: #.isDroneConnected:
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
            # self.connect_drone_button.setIcon(self.drone_icon_gray)

    # method to set the colored version of the icon
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
