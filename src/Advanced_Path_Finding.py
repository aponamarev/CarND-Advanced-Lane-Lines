#!/usr/bin/env python
"""Advanced_Path_Finding.py provides a class responsible for detection and visualization of lanes."""

__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"

import cv2, numpy as np
from Lane import Lane
from CameraCalibration import CameraCalibration as CC
from utils import abs_sobel_thresh, warper, map_window, fit_polynomial, plot_2nd_degree_polynomial, visualize_lanes

R = lambda M, y: (1+(2*M[0]*y+M[1])**2)**(2/3) / np.abs(2*M[0])

class APF(object):
    """Advanced_Path_Finding.py provides a class responsible for detection and visualization of lanes."""
    def __init__(self, calibration_file_path = 'calibration.p', sobel_kernel=3, sobel_threshold=(20,100)):
        self._kernel = sobel_kernel
        self._thresh = sobel_threshold
        self._center_shift = 40
        self._vertical_shift = 85
        self._x_scaller = 3.7 / 580
        self._y_scaller = 8.5 * 3 / 720
        self._window_width = 50
        self._window_height = 80
        self._search_margin = 100
        self._cc = CC(calibration_file_path)
        self._left_lane = Lane("_left_lane")
        self._right_lane = Lane("_right_lane")

    def detect(self, img):
        img = self._cc.undistort(img)
        # Detect edges
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        binary_img = abs_sobel_thresh(gray, thresh=self._thresh, kernel=self._kernel)
        binary_warped_img, (src, dst) = warper(binary_img,
                                               top_c_shift=self._center_shift,
                                               top_v_shift=self._vertical_shift)

        # find curvature of the lanes
        mapped_binary, centroids = map_window(binary_warped_img, self._window_width, self._window_height, self._search_margin)

        self._left_lane.current_fit = fit_polynomial(centroids[:, 0], binary_warped_img.shape[0])
        self._right_lane.current_fit = fit_polynomial(centroids[:, 1], binary_warped_img.shape[0])
        # create curves describing lanes
        self._left_lane.allx, self._left_lane.ally = plot_2nd_degree_polynomial(self._left_lane.current_fit, binary_warped_img.shape[0])
        self._right_lane.allx, self._right_lane.ally = plot_2nd_degree_polynomial(self._right_lane.current_fit, binary_warped_img.shape[0])
        # Find fit in real-world coordinates
        left_fit = fit_polynomial(centroids[:, 0], binary_warped_img.shape[0],
                                  x_scaller=self._x_scaller,
                                  y_scaller=self._y_scaller)
        right_fit = fit_polynomial(centroids[:, 1], binary_warped_img.shape[0],
                                   x_scaller=self._x_scaller,
                                   y_scaller=self._y_scaller)
        self._left_lane.radius_of_curvature = R(left_fit, self._left_lane.ally).mean()
        self._right_lane.radius_of_curvature = R(right_fit, self._right_lane.ally).mean()
        #self._left_lane.line_base_pos = binary_warped_img.shape[1]/2-np.mean(centroids[0])
        self._left_lane.line_base_pos = centroids[0][0]
        self._right_lane.line_base_pos = centroids[0][1]
        car_position = (binary_warped_img.shape[1] - self._left_lane.line_base_pos - self._right_lane.line_base_pos) / 2
        binary_filled = visualize_lanes(binary_warped_img, self._left_lane.ally, self._left_lane.allx, self._right_lane.allx)
        unwarp_binary, (src, dst) = warper(binary_filled, src=dst, dst=src)
        mapped_img = cv2.addWeighted(img, 1, unwarp_binary, 0.3, 0)
        cv2.putText(mapped_img, "Radius of curvature: {}m".format((self._left_lane.radius_of_curvature + self._right_lane.radius_of_curvature) // 2),
                    (5, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), thickness=1)
        cv2.putText(mapped_img, "Vehicle is {:.1f}m {} of center".format(np.abs(car_position * self._x_scaller),
                                                                         "left" if car_position < 0 \
                                                                             else "right"),
                    (int(mapped_img.shape[1] / 2), 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), thickness=1)
        return mapped_img, (self._left_lane, self._right_lane)


def main():

    from matplotlib import pyplot as plt

    test_img_path = "test_images/test3.jpg"
    apf = APF(calibration_file_path='src/calibration.p')

    img = cv2.cvtColor(cv2.imread(test_img_path), cv2.COLOR_BGR2RGB)

    mapped_img, (left, right) = apf.detect(img)

    f, p = plt.subplots(1,2, figsize=[20,6])
    p[0].set_title("original")
    p[0].imshow(img)
    p[1].set_title("mappedout")
    p[1].imshow(mapped_img)
    f.show()




    return True

if __name__=="__main__":
    main()

