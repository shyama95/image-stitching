import sys
from PyQt4.QtGui import *
from PyQt4.QtCore import *
import cv2
import numpy as np

import matplotlib.pyplot as plt

# The GUI structure definition is provided in gui.py
from gui import *

from imageStitchingFns import *


# class ImageStitching implements the GUI main window class
class ImageStitchingClass(QMainWindow):
    # stores the input images
    currentImageLeft = [0]
    currentImageRight = [0]

    # stores output image
    outputImage = [0]

    # GUI initialization
    def __init__(self, parent=None):
        super(ImageStitchingClass, self).__init__()
        QWidget.__init__(self, parent)
        self.ui = ImageStitching_UI()
        self.ui.setupUi(self)

        # Assigning functions to be called on all button clicked events and
        self.ui.openImageLeftButton.clicked.connect(lambda: self.open_image(isLeft=True))
        self.ui.openImageRightButton.clicked.connect(lambda: self.open_image(isLeft=False))
        self.ui.saveButton.clicked.connect(lambda: self.save_image())
        self.ui.resetButton.clicked.connect(lambda: self.reset_output())

        self.ui.sticthButton.clicked.connect(lambda: self.stitch())
        self.ui.viewSIFTButton.clicked.connect(lambda: self.view_sift_keypoints())
        # self.ui.viewSIFT2Button.clicked.connect(lambda: self.view_sift_keypoints_own())
        self.ui.toggleButton.clicked.connect(lambda: self.toggle_inputs())

    def stitch(self):
        zero = np.array([0])

        if not (np.array_equal(self.currentImageLeft, zero) or np.array_equal(self.currentImageRight, zero)):
            (status, result) = stitch_images(self.currentImageLeft, self.currentImageRight)

            if status:
                self.outputImage = result
                self.display_output_image()

    def view_sift_keypoints(self):
        zero = np.array([0])

        if not (np.array_equal(self.currentImageLeft, zero) or np.array_equal(self.currentImageRight, zero)):
            keypoint_left = get_sift_keypoints(self.currentImageLeft)
            keypoint_right = get_sift_keypoints(self.currentImageRight)

            keypoint_left_image = np.zeros_like(self.currentImageLeft)
            keypoint_right_image = np.zeros_like(self.currentImageRight)
            keypoint_left_image = cv2.drawKeypoints(self.currentImageLeft, keypoint_left, keypoint_left_image)
            keypoint_right_image = cv2.drawKeypoints(self.currentImageRight, keypoint_right, keypoint_right_image)

            # Plot input images with keypoints marked
            fig = plt.figure(figsize=(1, 2))

            fig.add_subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(keypoint_left_image, cv2.COLOR_BGR2RGB))
            plt.title('Input image left')

            fig.add_subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(keypoint_right_image, cv2.COLOR_BGR2RGB))
            plt.title('Input image right')

            plt.show()

    def view_sift_keypoints_own(self):
        zero = np.array([0])

        if not (np.array_equal(self.currentImageLeft, zero) or np.array_equal(self.currentImageRight, zero)):
            keypoint_left = get_sift_keypoints2(self.currentImageLeft)
            keypoint_right = get_sift_keypoints2(self.currentImageRight)

            keypoint_left_image = np.zeros_like(self.currentImageLeft)
            keypoint_right_image = np.zeros_like(self.currentImageRight)
            keypoint_left_image = cv2.drawKeypoints(self.currentImageLeft, keypoint_left, keypoint_left_image)
            keypoint_right_image = cv2.drawKeypoints(self.currentImageRight, keypoint_right, keypoint_right_image)

            # Plot input images with keypoints marked
            fig = plt.figure(figsize=(1, 2))

            fig.add_subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(keypoint_left_image, cv2.COLOR_BGR2RGB))
            plt.title('Input image left')

            fig.add_subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(keypoint_right_image, cv2.COLOR_BGR2RGB))
            plt.title('Input image right')

            plt.show()

    # toggle between left and right input images
    def toggle_inputs(self):
        temp = self.currentImageLeft
        self.currentImageLeft = self.currentImageRight
        self.currentImageRight = temp

        self.display_input_image(True)
        self.display_input_image(False)

    # implements image open functionality
    def open_image(self, isLeft):
        window_label = 'Left' if isLeft else 'Right'
        # open a new Open Image dialog box and capture path of file selected
        open_image_window = QFileDialog()
        image_path = QFileDialog.getOpenFileName\
            (open_image_window, 'Open ' + window_label + ' Image', '/')

        # check if image path is not null or empty
        if image_path:
            # read image at selected path to a numpy ndarray object as color image
            inputImage = cv2.imread(image_path)

            if isLeft:
                self.currentImageLeft = inputImage
                self.display_input_image(isLeft=True)
            else:
                self.currentImageRight = inputImage
                self.display_input_image(isLeft=False)

    # called when Save button is clicked
    def save_image(self):
        # configure the save image dialog box to use .jpg extension for image if
        # not provided in file name
        dialog = QFileDialog()
        dialog.setDefaultSuffix('jpg')
        dialog.setAcceptMode(QFileDialog.AcceptSave)

        # open the save dialog box and wait until user clicks 'Save'
        # button in the dialog box
        if dialog.exec_() == QDialog.Accepted:
            # select the first path in the selected files list as image save
            # location
            save_image_filename = dialog.selectedFiles()[0]
            # write current image to the file path selected by user
            cv2.imwrite(save_image_filename, self.outputImage)

    # Reset the output image
    def reset_output(self):
        self.ui.outputImageDisplay.clear()

    # display_input_image converts input image from ndarry format to pixmap and
    # assigns it to image display label
    def display_input_image(self, isLeft):
        # set display size to size of the image display label
        display_size = self.ui.imageLeftDisplay.size() if isLeft else self.ui.imageRightDisplay.size()
        # copy current image to temporary variable for processing pixmap
        image = np.array(self.currentImageLeft) if isLeft else np.array(self.currentImageRight)
        zero = np.array([0])

        (imageWidth, imageHeight) = (image.shape[1], image.shape[0])

        # display image if image is not [0] array
        if not np.array_equal(image, zero):
            # convert BGR image to RGB format for display in label
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # ndarray cannot be directly converted to QPixmap format required
            # by image display label
            # so ndarray is first converted to QImage and then QImage to QPixmap
            # convert image ndarray to QImage format
            qImage = QImage(image, imageWidth, imageHeight,
                            imageWidth * 3, QImage.Format_RGB888)

            # convert QImage to QPixmap for loading in image display label
            pixmap = QPixmap()
            QPixmap.convertFromImage(pixmap, qImage)
            pixmap = pixmap.scaled(display_size, Qt.KeepAspectRatio,
                                   Qt.SmoothTransformation)

            # set pixmap to image display label in GUI
            if isLeft:
                self.ui.imageLeftDisplay.setPixmap(pixmap)
            else:
                self.ui.imageRightDisplay.setPixmap(pixmap)

    # display_output_image converts output image from ndarry format to pixmap and
    # assigns it to image display label
    def display_output_image(self):
        # set display size to size of the image display label
        display_size = self.ui.outputImageDisplay.size()
        # copy current image to temporary variable for processing pixmap
        image = np.array(self.outputImage)
        zero = np.array([0])

        (imageWidth, imageHeight) = (image.shape[1], image.shape[0])

        # display image if image is not [0] array
        if not np.array_equal(image, zero):
            # convert BGR image to RGB format for display in label
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # ndarray cannot be directly converted to QPixmap format required
            # by image display label
            # so ndarray is first converted to QImage and then QImage to QPixmap
            # convert image ndarray to QImage format
            qImage = QImage(image, imageWidth, imageHeight,
                            imageWidth * 3, QImage.Format_RGB888)

            # convert QImage to QPixmap for loading in image display label
            pixmap = QPixmap()
            QPixmap.convertFromImage(pixmap, qImage)
            pixmap = pixmap.scaled(display_size, Qt.KeepAspectRatio,
                                   Qt.SmoothTransformation)

            # set pixmap to image display label in GUI
            self.ui.outputImageDisplay.setPixmap(pixmap)


# initialize the ImageStitchingClass and run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myapp = ImageStitchingClass()
    myapp.show()  # showMaximized()
    sys.exit(app.exec_())
