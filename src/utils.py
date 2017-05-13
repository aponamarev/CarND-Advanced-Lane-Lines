#!/usr/bin/env python
"""CameraCalibration.py is class that provides camera calibration tools."""

__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"

import numpy as np
import cv2

def abs_sobel_thresh(img, orient='x', kernel=3, thresh=(0,255)):
    """
    Transforms img into a binary mask based on Sobel kernel and provided threshold.
    :param img: input image (1 channel)
    :param orient: orientation of Sobel gradient, eith 'x' or 'y'
    :param kernel: the size of Sobel kernel (the larger the size the smother the gradient)
    :param thresh: condition used to transform the gradient into a binary mask
    :return: one channel image
    """
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel))
    elif orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel))
    else:
        raise AttributeError("Incorrect orient attribute. orient attribute accepts only 'x' and 'y'")
    # Rescale back to 8 bit integer
    sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    mask = np.zeros_like(sobel)
    mask[(sobel >= thresh[0]) & (sobel <= thresh[1])] = 1
    return mask

def mag_thresh(img, kernel=3, thresh=(0, 255)):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def dir_threshold(img, kernel=3, thresh=(0, np.pi / 2)):
    """
    Transforms an inputl image into a binary mask based on the direction of the gradient (set in radians).
    :param img: input image (1 channel)
    :param kernel: the size of Sobel kernel (the larger the size the smother the gradient)
    :param thresh: condition used to transform the gradient direction (in radians) into a binary mask
    :return: one channel image (binary mask)
    """
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    mask = np.zeros_like(absgraddir)
    mask[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return np.uint8(mask)