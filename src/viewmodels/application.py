import sys
# import some PyQt5 modules
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer, Qt
from PyQt5 import QtGui
import cv2 # import Opencv module
import pygame
sys.path.insert(0, r'C:\Users\Tavershima\source\repos\RemoteCropDisease\src\UI')
from ui_main_window import Ui_Form
from aircraft import Aircraft

# Speed of the drone
S = 60

class MainWindow(QWidget):
    # class constructor
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # initialize the aircraft which we will be communicating to
        self.aircraft = Aircraft()

        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.execute)
        self.timer.start(20)
        
        # set control_bt callback clicked  function
        #self.ui.control_bt.clicked.connect(self.controlTimer)

        # initialize a default. Its uncertain if any joystick is available so we assign None
        self.joystick = None
        pygame.init() # Init pygame to enable joystick and media objects
        
    def joystick_handler(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    self.aircraft.update()
                elif event.type == pygame.JOYBUTTONDOWN:
                    self.keyPressed(event.button)
                elif event.type == pygame.JOYBUTTONUP:
                    self.keyReleased(event.button)
                    
                if event.type == pygame.JOYDEVICEADDED:
                    # This event will be generated when the program starts for every
                    # joystick, filling up the list without needing to create them manually.
                    joy = pygame.joystick.Joystick(event.device_index)
                    self.joystick = joy
                elif event.type == pygame.JOYDEVICEREMOVED:
                    self.joystick = None

    def execute(self):
        for event in pygame.event.get():
            if event.type == pygame.USEREVENT + 1:
                self.aircraft.update()
            elif event.type == pygame.JOYBUTTONDOWN:
                self.keyPressed(event.button)
            elif event.type == pygame.JOYBUTTONUP:
                self.keyReleased(event.button)
            elif event.type == pygame.JOYDEVICEADDED:
                # This event will be generated when the program starts for any
                # joystick, automatically detecting it without needing to create it manually.
                self.joystick = pygame.joystick.Joystick(event.device_index)
            elif event.type == pygame.JOYDEVICEREMOVED:
                self.joystick = None

        frame = self.aircraft.get_frame().frame # get a frame from the aircraft
        qImg = self.convert_cv_qt(frame) # convert the image frame to QPixmap
        self.ui.image_label.setPixmap(qImg) # display the image on the UI

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""

        # optional: we can remove these two lines later
        text = "Battery: {}%".format(self.aircraft.get_battery())
        cv2.putText(cv_img, text, (5, 720 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(1080, 720, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            self.timer.start(20) # start timer
            self.ui.control_bt.setText("Stop") # update control_bt text
        else: # if timer is started
            self.timer.stop() # stop timer
            self.ui.control_bt.setText("Start") # update control_bt text

    def keyPressed(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key：an integer value identifying the joystick keyId or axis keyId that was pressed.
        """
        # attempt takeoff is not already
        if not self.aircraft.initite_takeoff():
            return

        if key == 0: # Triangle key
            self.aircraft.move(forward_back=S)  # set forward velocity
        elif key == 1: # Circle key
            self.aircraft.move(left_right=S)  # set right velocity
        elif key == 2: # Times key
            self.aircraft.move(forward_back=-S) # set backward velocity
        elif key == 3: # Square key
            self.aircraft.move(left_right=-S)  # set left velocity
        elif key == 4: # Left 1
            self.aircraft.move(up_down=S)  # set up velocity
        elif key == 5: # Right 1 key
            self.aircraft.move(yaw=S)  # set yaw right velocity
        elif key == 6: # Left 2 key
            self.aircraft.move(up_down=-S)  # set down velocity
        elif key == 7: # Right 2 key
            self.aircraft.move(yaw=-S)  # set yaw left velocity
        elif key == 10: # left steer button
            self.aircraft.stream_video() # start streming video
        elif key == 11: # Right steer button
            self.aircraft.capture_image() # take a snapshot
        
    def keyReleased(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key：an integer value identifying the joystick keyId or axis keyId that was pressed.
        """
        
        if key == 0: # Triangle key
            self.aircraft.move()  # set forward velocity
        elif key == 1: # Circle key
            self.aircraft.move()  # set right velocity
        elif key == 2: # Times key
            self.aircraft.move() # set backward velocity
        elif key == 3: # Square key
            self.aircraft.move()  # set left velocity
        elif key == 4: # Left 1
            self.aircraft.move()  # set up velocity
        elif key == 5: # Right 1 key
            self.aircraft.move()  # set yaw right velocity
        elif key == 6: # Left 2 key
            self.aircraft.move()  # set down velocity
        elif key == 7: # Right 2 key
            self.aircraft.move()  # set yaw left velocity
        elif key == 8:  # select key
            self.aircraft.initite_land()
        elif key == 9:  # start key
            self.aircraft.initite_takeoff()
        

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())