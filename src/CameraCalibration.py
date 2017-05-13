#!/usr/bin/env python
"""CameraCalibration.py is class that provides camera calibration tools."""

__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"


import pickle, cv2
from os.path import exists, join
from os import listdir
import numpy as np

class CameraCalibration(object):
    """Class provides camera calibration tools designed to calculated camera distortion and adjust images based on the
    camera matrix and coefficients. 
    """
    def __init__(self, path_to_callibration_file, nx=9, ny=7):
        self._nx = nx
        self._ny = ny
        # Ensure that provided path is correct
        if exists(path_to_callibration_file):
            # Load calibration file
            with open(path_to_callibration_file, mode='rb') as f:
                callibration_file = pickle.load(f)
            # Update callibration coefficients and camera matrix
            self._M = callibration_file["matrix"]
            self._distort_coeff = callibration_file["coefficients"]
        else:
            self._path_to_save = path_to_callibration_file
            self._M = None
            self._distort_coeff = None
            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            self._objp = np.zeros((nx * ny, 3), np.float32)
            self._objp[:, :2] = np.mgrid[0:self._nx, 0:self._ny].T.reshape(-1, 2)




    def callibrate(self, imgs_path):
        """Calculates and stores calibration matrix and distortion coefficients according to OpenCV description.
        Reference: Camera Calibration — OpenCV 3.0.0-dev documentation. 2017. Camera Calibration — 
        OpenCV 3.0.0-dev documentation. [ONLINE] Available at: 
        http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html. 
        [Accessed 11 May 2017]."""
        objps, imgps = [],[]
        for p in listdir(imgs_path):
            try:
                img = cv2.imread(join(imgs_path,p))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(gray, (self._nx, self._ny), None)
                # If found, add object points, image points (after refining them)
                if ret:
                    objps.append(self._objp)
                    imgps.append(corners)
                else:
                    print("Open CV wasn't able to identify chessboard grid in the following image: {}".format(p))

            except:
                print("Cannot open image at path {}".format(p))

        assert len(objps)>0, "Error: no chessboard grid points were captured"

        ret, self._M, self._distort_coeff, rvecs, tvecs = cv2.calibrateCamera(objps, imgps, gray.shape[::-1], None, None)
        with open(self._path_to_save, 'wb') as f:
            pickle.dump({"matrix": self._M, "coefficients": self._distort_coeff}, f)


    def undistort(self, img, alpha=0):
        """Returns undistorted image calibrated for camera calibration Matrix and distortion coefficients.
        By default - removes unwanted pixels. To recover full image with black pixes filled in for missing information
        set alpha=1.
        Reference: Camera Calibration — OpenCV 3.0.0-dev documentation. 2017. Camera Calibration — 
        OpenCV 3.0.0-dev documentation. [ONLINE] Available at: 
        http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html. 
        [Accessed 11 May 2017]."""
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self._M, self._distort_coeff, (w, h), alpha, (w, h))
        # undistort
        return cv2.undistort(img, self._M, self._distort_coeff, None, newcameramtx)

def main():
    from matplotlib import pyplot as plt
    # Define the location of files
    path_to_chess_images = "../camera_cal"
    path_to_callibration_file = "calibration.p"

    # Define the grid
    nx = 9
    ny = 6

    camera = CameraCalibration(path_to_callibration_file, nx, ny)
    #camera.callibrate(path_to_chess_images)

    for p in listdir(path_to_chess_images):
        try:
            img = cv2.cvtColor(cv2.imread(join(path_to_chess_images, p)), cv2.COLOR_BGR2RGB)
            f, (im1, im2) = plt.subplots(1,2)
            im1.set_title("original")
            im1.imshow(img)
            im2.set_title("undistorted")
            im2.imshow(camera.undistort(img))
            print("Image {} was processed successfully.".format(p))
        except:
            print("Incorrect image path: {}".format(p))


if __name__=="__main__":
    main()
