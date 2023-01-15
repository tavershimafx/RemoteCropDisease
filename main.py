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
import time
import threading
from PySide6 import QtCore
from PySide6.QtCore import Qt, QThread, Signal, Slot, QSize, QTimer
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap, QIcon, QColor
from PySide6 import QtWidgets, QtSvg, QtGui
from widgets import CustomDialog
import pandas as pd
from pathlib import Path
from datetime import datetime
from constants import CLASSES
import multiprocessing
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QSlider,
    QScrollArea,
)
from detection_utils import *
from viewmodels.aircraft import Aircraft
import pygame
from os import environ
import uuid

# pictures_folder = os.path.join(environ["USERPROFILE"], "Pictures", "tello")
desktop = os.path.expanduser("~/Desktop")
# get the current date time
current_dateTime = datetime.now()
filepath = Path(f"{desktop}/leaf Disease Detection/predictions/pictures")
filepath.parent.mkdir(parents=True, exist_ok=True)
pictures_folder = f"{desktop}/leaf Disease Detection/predictions/pictures"
# qapp = QtWidgets.qApp

# Speed of the drone
S = 60

# image saving time gap
SAVE_TIME_GAP = 20

TFLITE_MODEL_PATH = "resources/cropdisease.tflite"
MODEL_INPUT_SIZE = 224
INPUT_DIM = (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)

DEFAULT_MIN_CLASS = 0.8
DEFAULT_MIN_BOX = 0.6

# initialize the default number of threads
NUMBER_OF_THREADS = 2
# try to optimize for the number of threads on the computer
cpu_count = multiprocessing.cpu_count()
NUMBER_OF_THREADS = cpu_count
# print(type(cpu_count))


def efficient_lite(
    img,
    detection_threshold,
    class_threshold,
    isPredict,
    isCassava,
):
    # Load the TFLite model
    options = ObjectDetectorOptions(
        num_threads=1,
        score_threshold=detection_threshold,
        class_threshold=class_threshold,
    )
    detector = ObjectDetector(model_path=TFLITE_MODEL_PATH, options=options)

    # Run object detection estimation using the model.
    detections = detector.detect(img)

    # Draw keypoints and edges on input image
    # image_np, predictions = visualize(img, detections)

    # Draw keypoints and edges on input image using classnames predicted by mobilent
    image_np, predictions = visualize_classnames_with_mobilenet(
        img,
        detections,
        class_threshold,
        isPredict,
        isCassava,
    )
    return image_np, predictions


def save_images_periodically(img):
    # # get the current date time

    if not os.path.exists(pictures_folder):
        os.mkdir(pictures_folder)

    cv2.imwrite(f"{pictures_folder}/{uuid.uuid4().hex}.png", img)

    # threading.Timer(SAVE_TIME_GAP, save_images_periodically, [img]).start()


class Thread(QThread):
    updateFrame = Signal(QImage)
    updateStatus = Signal(bool)
    prediction_dict = Signal(dict)

    def __init__(self, no_wifi, parent=None):
        QThread.__init__(self, parent)
        self.trained_file = None
        self.status = True
        self.no_wifi = no_wifi
        self.isConnected = True

        self.isPredict = False
        self.isFocus = False
        self.isCassava = False
        self.minClass = DEFAULT_MIN_CLASS
        # if the model is not minProbability sure about a prediction it shouldn't return
        # or draw the prediction
        self.minProbability = DEFAULT_MIN_BOX
        # list of the predcitions gotten from the frames
        self.predictions = []

        # self.save_img_routine = threading.Timer(SAVE_TIME_GAP, save_images_periodically).start()
        self.aircraft = Aircraft()

        # self.no_connection_image = no_connection_image.toImage()

    def set_minClass(self, minClass):
        self.minClass = float(minClass)

    def set_minProbability(self, probability):
        self.minProbability = float(probability)

    def set_aircraft(self, aircraft):
        self.aircraft = aircraft

    def set_crop(self):
        self.isCassava = not self.isCassava

    def start_stop_predictions(self):
        self.isPredict = not self.isPredict

    def set_focus(self):
        self.isFocus = not self.isFocus

    def run(self):
        """
        This function extracts the frames one by one
        from the self.cap attribute and process them
        detecting and drawing bounding boxes on the frames
        and emiting each frame to the MainWindow
        """
        # self.aircraft = Aircraft()
        counter = 0
        while self.status:
            # try and get a state. if it throws an error, the drone is no longer connected

            counter += 1
            # get a frame from the aircraft
            try:
                frame_read = self.aircraft.get_frame()
                frame = frame_read.frame
                print("FRAME HAS BEEN READ FROM AIRCRAFT")
                # this happens when we lost the video feed from the drone

                if self.isPredict:
                    img_detections, predictions = efficient_lite(
                        frame,
                        self.minProbability,
                        self.minClass,
                        self.isPredict,
                        self.isCassava,
                    )
                    for prediction in predictions:
                        self.prediction_dict.emit(prediction)

                    # 🥸 AKO JOGODO abeg help me comment this line, try the next one make i see wetin go happen
                    color_frame = cv2.cvtColor(img_detections, cv2.COLOR_BGR2RGB)

                    # if counter % 30 == 0:
                    # Thread(save_images_periodically, [frame]).start()
                    save_images_periodically(frame)

                    # Creating and scaling QImage
                    h, w, ch = color_frame.shape
                    img = QImage(color_frame.data, w, h, ch * w, QImage.Format_RGB888)
                    scaled_img = img.scaled(1200, 600, Qt.KeepAspectRatio)
                    # # change self.predict back to false
                    # if len(predictions) != 0:
                    #     self.isPredict = False
                    # Emit signal

                    self.updateFrame.emit(scaled_img)

                else:
                    color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = color_frame.shape
                    img = QImage(color_frame.data, w, h, ch * w, QImage.Format_RGB888)
                    scaled_img = img.scaled(1200, 600, Qt.KeepAspectRatio)
                    self.updateFrame.emit(scaled_img)
                self.updateStatus.emit(True)
            except Exception as e:
                print("Could not connect to the drone")
                print(e)
                self.aircraft = None
                self.aircraft = Aircraft()
                self.isConnected = False
                # self.aircraft.tello.cap.release()
                self.updateFrame.emit(self.no_wifi)
                self.updateStatus.emit(False)
                break

        # self.exit(-1)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Title and dimensions
        self.setWindowTitle("Leaf Disease Detection")
        self.setGeometry(0, 0, 800, 700)

        # Drone control variables
        self.isStart = False
        self.isPredict = False
        self.isFocus = False
        self.isCassava = False

        self.minClassification = DEFAULT_MIN_CLASS
        self.minProbability = DEFAULT_MIN_BOX

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
        self.no_wifi = self.pixmap.toImage()
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
        self.th = Thread(self.no_wifi, parent=self)
        self.th.finished.connect(self.close)
        self.th.updateFrame.connect(self.setImage)
        self.th.prediction_dict.connect(self.updatePredictionList)
        self.th.updateStatus.connect(self.setDroneStatus)
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
        self.thresholdSlider1 = QSlider(orientation=Qt.Orientation.Horizontal)
        self.areaSlider.setMinimum(100)
        self.areaSlider.setMaximum(1000)
        self.areaSlider.setTickInterval(50)
        # threshold slider for bounding box model
        self.thresholdSlider.setMinimum(0)
        self.thresholdSlider.setMaximum(100)
        self.thresholdSlider.setTickInterval(10)

        # threshold slider for classification model
        self.thresholdSlider1.setMinimum(0)
        self.thresholdSlider1.setMaximum(100)
        self.thresholdSlider1.setTickInterval(10)

        top_slider_layout.addWidget(QLabel("Class Threshold"), 5)
        self.threshold1Label = QLabel(f"{self.minClassification}")
        self.threshold1MinLabel = QLabel("         0")
        self.threshold1MaxLabel = QLabel("1")
        top_slider_layout.addWidget(self.threshold1Label, 2)
        top_slider_layout.addWidget(self.threshold1MinLabel, 5)
        top_slider_layout.addWidget(self.thresholdSlider1, 83)
        top_slider_layout.addWidget(self.threshold1MaxLabel, 5)

        self.thresholdLabel = QLabel(f"{self.minProbability}")
        self.thresholdMinLabel = QLabel("         0")
        self.thresholdMaxLabel = QLabel("1")

        self.thresholdSlider1.setValue(self.minClassification)
        self.thresholdSlider.setValue(self.minProbability)
        bottom_slider_layout.addWidget(QLabel("Bounding Box Threshold"), 2)
        bottom_slider_layout.addWidget(self.thresholdLabel, 2)
        bottom_slider_layout.addWidget(self.thresholdMinLabel, 2)
        bottom_slider_layout.addWidget(self.thresholdSlider, 88)
        bottom_slider_layout.addWidget(self.thresholdMaxLabel, 5)

        # model_layout.addLayout(top_slider_layout, 1)
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
        # check box button for cassava
        self.cassava_button = QPushButton()
        self.cassava_button.setText("NOT CASSAVA")
        # self.cassava_button.setCheckState(Qt.CheckState.Checked)

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
        self.start_button_text = "FOCUS"
        self.start_stop_button = QPushButton(self.start_button_text)
        self.predict_button = QPushButton("PREDICT")
        start_predict_button_layout.addWidget(self.start_stop_button)
        start_predict_button_layout.addWidget(self.predict_button)

        left_layout = QVBoxLayout()
        left_layout.addLayout(buttons_layout)
        left_layout.addWidget(self.velocity_altitude_group, 1)
        # left_layout.addWidget(self.altitude_group, 1)
        left_layout.addWidget(self.cassava_button, 1)
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
        # connect all buttons here
        self.connect_drone_button.clicked.connect(self.connect_drone)
        self.predict_button.clicked.connect(self.predict_start_stop)
        self.export_to_csv_button.clicked.connect(self.export_to_csv)
        self.thresholdSlider.valueChanged.connect(self.thresholdChange)
        self.thresholdSlider1.valueChanged.connect(self.threshold1Change)
        self.cassava_button.clicked.connect(self.change_crop)
        self.start_stop_button.clicked.connect(self.focus)
        self.setNoWifi()

        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.joystick_handler)
        self.timer.start(5)

        # self.start()

    @Slot()
    def set_model(self, text):
        self.th.set_file(text)

    @Slot()
    def change_crop(self):
        self.drone_not_connected = CustomDialog(
            title="Error",
            content="Drone is not connected. Please make sure you connect to the drone's wifi",
        )
        if self.th.aircraft.is_connected:
            if self.isCassava:
                self.cassava_button.setText("NOT CASSAVA")
                self.isCassava = False
                self.th.set_crop()
                # call the function to change the prediction model
            else:
                self.cassava_button.setText("CASSAVA")
                self.isCassava = True
                self.th.set_crop()
                # call the function to change the prediction model
        else:
            self.drone_not_connected.exec()

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

    @Slot()
    def setDroneStatus(self, status):
        if status:
            self.connect_drone_button.setIcon(self.drone_icon_green)
            # self.start_stop_button.setText("CONNECTED")
        else:
            self.connect_drone_button.setIcon(self.drone_icon_gray)
            # self.start_stop_button.setText("DISCONNECTED")

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
    def threshold1Change(self):
        """
        This method handles the changes to the areavalue when the area slider is moved
        """
        # area is a value between 500 and 1000
        # get the value
        minClass = self.thresholdSlider1.value()
        self.minClassification = minClass / 100
        # set the value on the label text
        self.threshold1Label.setText(f"{self.minClassification}")
        # set change the area value on the predicton thread
        self.th.set_minClass(self.minClassification)

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
            # try reintializing the drone object before connecting
            # self.th.set_aircraft(Aircraft())
            self.th.aircraft.connect()
            self.connect_drone_button.setIcon(self.drone_icon_green)
            # self.start_stop_button.setText("CONNECTED")
            self.drone_connection_success.exec()
            self.th.start()
        except:
            # self.start_stop_button.setText("DISCONNECTED")
            self.drone_connection_failed.exec()

    @Slot()
    def focus(self):
        """
        This method starts or stops the predictions
        """
        self.drone_not_connected = CustomDialog(
            title="Error",
            content="Drone is not connected. Please make sure you connect to the drone's wifi",
        )
        if self.th.aircraft.is_connected:  # .isDroneConnected:
            # start predictions
            if self.isFocus:
                # stop predictions
                self.isFocus = False
                self.th.set_focus()
                # set start stop button text to on
                self.start_stop_button.setText("NO FOCUS")
            else:
                # start the drone
                self.isFocus = True
                self.th.set_focus()
                # set start stop button text to off
                self.start_stop_button.setText("FOCUS")
        else:
            self.drone_not_connected.exec()
            self.drone_icon_color = QColor(255, 0, 0)

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
