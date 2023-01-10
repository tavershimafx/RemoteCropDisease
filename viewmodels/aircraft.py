from djitellopy import Tello
import cv2
import time
from os import environ
import os.path
import uuid
from datetime import datetime
from pathlib import Path
from threading import Thread
import time, threading

# pictures_folder = os.path.join(environ["USERPROFILE"], "Pictures", "tello")
desktop = os.path.expanduser("~/Desktop")
# get the current date time
current_dateTime = datetime.now()
filepath = Path(f"{desktop}/leaf Disease Detection/predictions/pictures")
filepath.parent.mkdir(parents=True, exist_ok=True)
pictures_folder = f"{desktop}/leaf Disease Detection/predictions/pictures"
# Speed of the drone
S = 60

# Frames per second of the pygame window display
# A low number also results in input lag, as input information is processed once per frame.
FPS = 180


class Aircraft(object):
    def __init__(self):
        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10
        self.has_takeoff = False

        self.send_rc_control = False

        self.is_connected = False
        self.is_streaming_video = False
        self.keepRecording = True

        self.frame_read = None

        # we need to run the recorder in a seperate thread, otherwise blocking options
        #  would prevent frames from getting added to the video
        self.recorder = Thread(
            target=self.videoRecorder, args=lambda: self.keepRecording
        )

    def connect(self):
        """
        Connect to the drone and activate parameters for streaming
        """
        self.tello.connect()
        self.tello.set_speed(self.speed)

        # In case streaming is on. This happens when we quit this program abruptly.
        self.tello.streamoff()
        self.tello.streamon()

        self.frame_read = self.tello.get_frame_read()

        self.is_connected = True

    def initite_land(self):
        """
        Land the aircraft
        """
        if self.is_connected and self.has_takeoff:
            not self.tello.land()  # land
            self.send_rc_control = False
            self.has_takeoff = False

    def initite_takeoff(self):
        """
        Attempts to put the aircraft in motion by taking off. If the operation fails, retries
        for 10 times, if the operation fails then quits.
        """
        retry_times = 0
        while self.is_connected and retry_times < 10 and not self.has_takeoff:
            try:
                self.tello.takeoff()  # takeoff
                self.send_rc_control = True
                self.has_takeoff = True  # make sure the aircraft has taken off before attempting to control it
                return True
            except:
                print("Error occurrd taking off. Retrying...")

        if self.has_takeoff:
            return True

        self.tello.send_keepalive()

        # if the aircraft retries takeoff for 10 times without any success, quit taking
        # off. Maybe the aircraft became unavailable or is busy
        if retry_times == 10 and not self.has_takeoff:
            print("aircraft attempted takeoff 10 times without success. Quitting...")
            return False

    def move(self, forward_back=0, left_right=0, up_down=0, yaw=0):
        """
        Updates the aircraft motion according to the velocities passed to it
        """

        self.back_velocity = forward_back
        self.left_right_velocity = left_right
        self.up_down_velocity = up_down
        self.yaw_velocity = yaw

        if self.is_connected and self.send_rc_control:
            self.tello.send_rc_control(left_right, forward_back, up_down, yaw)

    def update(self):
        """
        Update routine. Send velocities to Tello.
        """
        if self.is_connected and self.send_rc_control:
            self.tello.send_rc_control(
                self.left_right_velocity,
                self.back_velocity,
                self.up_down_velocity,
                self.yaw_velocity,
            )

    def capture_image(self):
        """
        Captures an image and save it in the Pictures folder
        """

        # create directory if it does not exist
        if not os.path.exists(pictures_folder):
            os.mkdir(pictures_folder)

        if self.is_connected:
            cv2.imwrite(
                f"{pictures_folder}/{uuid.uuid4().hex}.png", self.frame_read.frame
            )

    def stream_video(self):
        """
        Streams a video and save it in the Pictures folder
        """

        # create directory if it does not exist
        if not os.path.exists(pictures_folder):
            os.mkdir(pictures_folder)

        if self.is_connected:
            if not self.is_streaming_video:
                self.keepRecording = True
                self.recorder.start()
            else:
                self.is_streaming_video = False
                self.keepRecording = False
                self.recorder._stop_event.set()
                self.recorder.join()

    def get_frame(self):
        # self.send_keepalive()
        """Returns the frame handle used to capture images from the vehicle"""
        if self.frame_read:
            return self.frame_read
        else:
            return None

    def get_battery(self):
        """Returns the battery level of the aircraft"""
        if self.is_connected:
            return self.tello.get_battery()
        else:
            return 0

    def get_status(self):
        try:
            status = self.tello.get_battery()
            print(f"this is the status of the drone {status}")
        except:
            print("Could not get drone status cause it is not connected")

    def get_altitude(self):
        """Returns the current altitude of the aircraft"""
        if self.is_connected:
            return self.tello.get_height()
        else:
            return 0

    def videoRecorder(self, keepRecording):
        if self.is_connected:
            # create a VideoWrite object, recoring to ./guid.avi
            height, width, _ = self.frame_read.frame.shape
            # video = cv2.VideoWriter(f"{pictures_folder}/{uuid.uuid4().hex}.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
            video = cv2.VideoWriter(
                f"{pictures_folder}/{uuid.uuid4().hex}.mp4",
                cv2.VideoWriter_fourcc(*"MP4V"),
                30,
                (width, height),
            )

            while keepRecording:
                video.write(self.frame_read.frame)
                time.sleep(1 / 30)

            video.release()

    def get_state(self, key):
        return self.tello.get_state_field(key)
