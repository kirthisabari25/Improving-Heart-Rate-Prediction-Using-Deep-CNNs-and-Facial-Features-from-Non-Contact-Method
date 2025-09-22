import cv2
import numpy as np
from PyQt5 import QtCore

from PyQt5.QtCore import QThread
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import QPushButton, QApplication, QComboBox, QLabel, QFileDialog, QStatusBar, QDesktopWidget, QMessageBox, QMainWindow
import time
import pyqtgraph as pg
import sys

from process import Process
from webcam import Webcam
from video import Video
from interface import waitKey, plotXY

class GUI(QMainWindow, QThread):
    def __init__(self):
        super(GUI,self).__init__()
        self.initUI()
        self.webcam = Webcam()
        self.video = Video()
        self.input = self.webcam
        self.dirname = ""
        print("Input: webcam")
        self.statusBar.showMessage("Input: webcam", 5000)
        self.btnOpen.setEnabled(False)
        self.process = Process()
        self.status = False
        self.frame = np.zeros((10, 10, 3), np.uint8)
        self.bpm = 0
        self.terminate = False

    def initUI(self):
        # Set font
        font = QFont()
        font.setPointSize(16)

        # Widgets
        self.btnStart = QPushButton("Start", self)
        self.btnStart.move(440, 520)
        self.btnStart.setFixedWidth(200)
        self.btnStart.setFixedHeight(50)
        self.btnStart.setFont(font)
        self.btnStart.clicked.connect(self.run)

        self.btnOpen = QPushButton("Open", self)
        self.btnOpen.move(230, 520)
        self.btnOpen.setFixedWidth(200)
        self.btnOpen.setFixedHeight(50)
        self.btnOpen.setFont(font)
        self.btnOpen.clicked.connect(self.openFileDialog)

        self.cbbInput = QComboBox(self)
        self.cbbInput.addItem("Webcam")
        self.cbbInput.addItem("Video")
        self.cbbInput.setCurrentIndex(0)
        self.cbbInput.setFixedWidth(200)
        self.cbbInput.setFixedHeight(50)
        self.cbbInput.move(20, 520)
        self.cbbInput.setFont(font)
        self.cbbInput.activated.connect(self.selectInput)

        self.lblDisplay = QLabel(self)  # Label to show frame from camera
        self.lblDisplay.setGeometry(10, 10, 640, 480)
        self.lblDisplay.setStyleSheet("background-color: #000000")

        self.lblROI = QLabel(self)  # Label to show face with ROIs
        self.lblROI.setGeometry(660, 10, 200, 200)
        self.lblROI.setStyleSheet("background-color: #000000")

        self.lblHR = QLabel(self)  # Label to show HR change over time
        self.lblHR.setGeometry(900, 20, 300, 40)
        self.lblHR.setFont(font)
        self.lblHR.setText("Frequency: ")

        self.lblHR2 = QLabel(self)  # Label to show stable HR
        self.lblHR2.setGeometry(900, 70, 300, 40)
        self.lblHR2.setFont(font)
        self.lblHR2.setText("Heart rate: ")

        # Dynamic plot
        self.signal_Plt = pg.PlotWidget(self)
        self.signal_Plt.move(660, 220)
        self.signal_Plt.resize(480, 192)
        self.signal_Plt.setLabel('bottom', "Signal")

        self.fft_Plt = pg.PlotWidget(self)
        self.fft_Plt.move(660, 425)
        self.fft_Plt.resize(480, 192)
        self.fft_Plt.setLabel('bottom', "FFT")

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(200)

        self.statusBar = QStatusBar()
        self.statusBar.setFont(font)
        self.setStatusBar(self.statusBar)

        # Config main window
        self.setGeometry(100, 100, 1160, 640)
        self.setWindowTitle("Heart rate monitor")
        self.show()

    def update(self):
        self.signal_Plt.clear()
        if len(self.process.samples) > 20:
            self.signal_Plt.plot(self.process.samples[20:], pen='g')

        self.fft_Plt.clear()
        if len(self.process.freqs) > 0 and len(self.process.fft) > 0:
            self.fft_Plt.plot(np.column_stack((self.process.freqs, self.process.fft)), pen='g')

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Message", "Are you sure you want to quit",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.terminate = True
            if self.input:
                self.input.stop()
            event.accept()
        else:
            event.ignore()

    def selectInput(self):
        self.reset()
        if self.cbbInput.currentIndex() == 0:
            self.input = self.webcam
            print("Input: webcam")
            self.btnOpen.setEnabled(False)
        elif self.cbbInput.currentIndex() == 1:
            self.input = self.video
            print("Input: video")
            self.btnOpen.setEnabled(True)

    def key_handler(self):
        """
        cv2 window must be focused for keypresses to be detected.
        """
        self.pressed = waitKey(1) & 255  # wait for keypress for 10 ms
        if self.pressed == 27:  # exit program on 'esc'
            print("[INFO] Exiting")
            self.webcam.stop()
            sys.exit()

    def openFileDialog(self):
        self.dirname, _ = QFileDialog.getOpenFileName(self, 'Open File')

    def reset(self):
        self.process.reset()
        self.lblDisplay.clear()
        self.lblDisplay.setStyleSheet("background-color: #000000")

    def main_loop(self):
        frame = self.input.get_frame()
        if frame is None or frame.size == 0:
            print("[WARNING] Empty frame received.")
            return

        self.process.frame_in = frame
        if not self.terminate:
            ret = self.process.run()

        if ret:
            self.frame = self.process.frame_out  # Get the frame to show in GUI
            self.f_fr = self.process.frame_ROI  # Get the face to show in GUI
            self.bpm = self.process.bpm  # Get the bpm change over time
        else:
            self.frame = frame
            self.f_fr = np.zeros((10, 10, 3), np.uint8)
            self.bpm = 0

        if self.frame is not None and self.frame.size != 0:
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
            cv2.putText(self.frame, "FPS " + str(float("{:.2f}".format(self.process.fps))),
                        (20, 460), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
            img = QImage(self.frame, self.frame.shape[1], self.frame.shape[0],
                         self.frame.strides[0], QImage.Format_RGB888)
            self.lblDisplay.setPixmap(QPixmap.fromImage(img))

        if self.f_fr is not None and self.f_fr.size != 0:
            self.f_fr = cv2.cvtColor(self.f_fr, cv2.COLOR_RGB2BGR)
            f_img = QImage(self.f_fr, self.f_fr.shape[1], self.f_fr.shape[0],
                           self.f_fr.strides[0], QImage.Format_RGB888)
            self.lblROI.setPixmap(QPixmap.fromImage(f_img))

        self.lblHR.setText("Freq: " + str(float("{:.2f}".format(self.bpm))))

        if len(self.process.bpms) > 50:
            if len(self.process.bpms) > 0 and not np.isnan(self.process.bpms).any():
                mean_bpm = np.mean(self.process.bpms)
                if not np.isnan(mean_bpm) and len(self.process.bpms) > 1 and max(self.process.bpms - mean_bpm) < 5:
                    self.lblHR2.setText("Heart rate: " + str(float("{:.2f}".format(mean_bpm))) + " bpm")

        self.key_handler()

    def run(self):
        print("run")
        self.reset()
        self.input.dirname = self.dirname
        if self.input.dirname == "" and self.input == self.video:
            print("Choose a video first")
            return
        if not self.status:
            self.status = True
            self.input.start()
            self.btnStart.setText("Stop")
            self.cbbInput.setEnabled(False)
            self.btnOpen.setEnabled(False)
            self.lblHR2.clear()

            self.loop_timer = QtCore.QTimer(self)
            self.loop_timer.timeout.connect(self.main_loop)
            self.loop_timer.start(30)

        else:
            self.stop()

    def stop(self):
        if self.status:
            self.status = False
            self.input.stop()
            self.btnStart.setText("Start")
            self.cbbInput.setEnabled(True)
            if hasattr(self, 'loop_timer'):
                self.loop_timer.stop()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GUI()
    sys.exit(app.exec_())
