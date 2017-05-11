#!/usr/bin/env python
"""CameraCalibration.py is class that provides camera calibration tools."""

__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"


import pickle
from os.path import exists
from os import listdir
from matplotlib.image import imread

class CameraCalibration(object):
    """Class provides camera calibration tools designed to calculated camera distortion and adjust images based on the
    camera matrix and coefficients. 
    """
    def __init__(self, pickel_path = None):
        # Ensure that provided path is correct
        assert exists(pickel_path), "Incorrect path provided. {} file doesn't exist.".format(pickel_path)
        # Load calibration file
        callibration_file = pickle.load(pickel_path)
        # Update callibration coefficients and camera matrix
        self._M = callibration_file["matrix"]
        self._coefficients = callibration_file["coefficients"]


    def callibrate(self, imgs_path):
        chessboard_imgs = []
        for p in listdir(imgs_path):
            try:
                img = imread(p)
                chessboard_imgs.append(img)
            except:
                print("Cannot open image at path {}".format(p))

        

    def undistort(self, img):
        raise NotImplementedError

    def _loadPickle(self, path):
        raise NotImplementedError