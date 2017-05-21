#!/usr/bin/env python
"""Advanced_Path_Finding.py provides a class responsible for detection and visualization of lanes."""

__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"

import cv2, numpy as np
from matplotlib import pyplot as plt

if __name__=="__main__":
    from Lane import Lane
    from CameraCalibration import CameraCalibration as CC
    from utils import abs_sobel_thresh, mag_thresh, warper, map_window, find_window_centroids, \
        fit_polynomial, plot_2nd_degree_polynomial, visualize_lanes, dir_threshold
else:
    from .Lane import Lane
    from .CameraCalibration import CameraCalibration as CC
    from .utils import abs_sobel_thresh, mag_thresh, warper, map_window, find_window_centroids, \
        fit_polynomial, plot_2nd_degree_polynomial, visualize_lanes, dir_threshold

R = lambda M, y: (1+(2*M[0]*y+M[1])**2)**(2/3) / np.abs(2*M[0])

class APF(object):
    """Advanced_Path_Finding.py provides a class responsible for detection and visualization of lanes."""
    def __init__(self, calibration_file_path = 'calibration.p',
                 sobel_kernel=3, h_threshold=(15, 25), y_threshold=(200, 255),
                 sobel_window=(50,80,100), direction_threshold=(30*np.pi/180, 80*np.pi/180), frames=8):
        assert not sobel_window[0] % 2 == 0, "Error: Window size should be odd."
        self._kernel = sobel_kernel
        self._h_thresh = h_threshold
        self._y_thresh = y_threshold
        self._dir_thresh = direction_threshold
        self._center_shift = 40
        self._vertical_shift = 85
        self._x_scaller = 3.7 / 580
        self._y_scaller = 8.5 * 3 / 720
        self._window_width = sobel_window[0]
        self._window_height = sobel_window[1]
        self._search_margin = sobel_window[2]
        self._cc = CC(calibration_file_path)
        self._left_lane = Lane("_left_lane", frames=frames)
        self._right_lane = Lane("_right_lane")

    def detect(self, img, debug=False):
        img = self._cc.undistort(img)
        # Convert to a single layer
        warped_img, (src, dst) = warper(img, top_c_shift=self._center_shift, top_v_shift=self._vertical_shift)
        # Detect edges
        # 1. Define color filters
        gray = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
        s = cv2.cvtColor(warped_img, cv2.COLOR_RGB2HLS)[:,:,0]
        sb = np.zeros_like(s, dtype=np.uint8)
        sb[(s >= self._h_thresh[0]) & (s < self._h_thresh[1])] = 1
        y = cv2.cvtColor(warped_img, cv2.COLOR_RGB2YUV)[:, :, 0]
        yb = np.zeros_like(y, dtype=np.uint8)
        yb[(y >= self._y_thresh[0]) & (y < self._y_thresh[1])] = 1
        # 2. Overlay a direction filter
        dir_gradient = dir_threshold(gray,kernel=self._kernel, thresh=self._dir_thresh)
        # 3. Combine results of color and direction filters
        combined = np.zeros_like(gray, dtype=np.uint8)
        combined[((sb == 1) & (dir_gradient == 1)) | (yb == 1)] = 1
        # 4. Remove noise on the far left side of the image
        combined[:,:250] = 0

        # find curvature of the lanes
        try:
            # 1. Find centroids
            centroids = find_window_centroids(combined, self._window_width, self._window_height, self._search_margin)
            # 2. Get pixels for left and right lines associated with identified centroids
            l_pixels, r_pixels = map_window(combined, centroids, self._window_width, self._window_height)
            # 3. Calculate curve coefficients
            self._left_lane.current_fit = np.polyfit(l_pixels[0], l_pixels[1], 2)
            self._right_lane.current_fit = np.polyfit(r_pixels[0], r_pixels[1], 2)
            # draw curves describing lanes
            self._left_lane.allx, self._left_lane.ally = plot_2nd_degree_polynomial(self._left_lane.best_fit, warped_img.shape[0])
            self._right_lane.allx, self._right_lane.ally = plot_2nd_degree_polynomial(self._right_lane.best_fit, warped_img.shape[0])
            # Calculating radius of curvature
            # Find fit in real-world coordinates
            left_fit = np.polyfit(l_pixels[0]*self._y_scaller, l_pixels[1]*self._x_scaller, 2)
            right_fit = np.polyfit(r_pixels[0]*self._y_scaller, r_pixels[1]*self._x_scaller, 2)
            # Calucate radius
            self._left_lane.radius_of_curvature = R(left_fit, self._left_lane.ally).mean()
            self._right_lane.radius_of_curvature = R(right_fit, self._right_lane.ally).mean()
            # Assign coordinates of lines
            self._left_lane.line_base_pos = centroids[0][0]
            self._right_lane.line_base_pos = centroids[0][1]
        except:
            self._left_lane.current_fit = self._left_lane.best_fit
            self._right_lane.current_fit = self._right_lane.best_fit
        # estimate car position
        car_position = (warped_img.shape[1] - self._left_lane.line_base_pos - self._right_lane.line_base_pos) / 2
        # Visualize result
        binary_filled = visualize_lanes(combined, self._left_lane.ally, self._left_lane.allx, self._right_lane.allx)
        unwarp_binary, (src, dst) = warper(binary_filled, src=dst, dst=src)
        mapped_img = cv2.addWeighted(img, 1, unwarp_binary, 0.3, 0)
        cv2.putText(mapped_img, "Radius of curvature: {}m".format((self._left_lane.radius_of_curvature + self._right_lane.radius_of_curvature) // 2),
                    (5, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), thickness=1)
        cv2.putText(mapped_img, "Vehicle is {:.1f}m {} of center".format(np.abs(car_position * self._x_scaller),
                                                                         "left" if car_position < 0 \
                                                                             else "right"),
                    (int(mapped_img.shape[1] / 2), 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), thickness=1)

        if debug:
            # Visualize steps for debugging
            mapped_img_ = cv2.resize(mapped_img, (mapped_img.shape[1]//2, mapped_img.shape[0]//2))
            sb = cv2.merge((sb, sb, sb))*255
            sb = cv2.resize(sb, (sb.shape[1]//2, sb.shape[0]//2))
            yb = cv2.merge((yb,yb,yb))*255
            yb = cv2.resize(yb, (yb.shape[1]//2, yb.shape[0]//2))
            combined = cv2.merge((combined, combined, combined))*255
            combined = cv2.resize(combined, (combined.shape[1]//2, combined.shape[0]//2))

            mapped_img = np.zeros_like(img, dtype=np.uint8)

            mapped_img[:sb.shape[0],:sb.shape[1]] = sb
            mapped_img[:yb.shape[0], yb.shape[1]:] = yb
            mapped_img[combined.shape[0]:, :combined.shape[1]] = combined
            mapped_img[mapped_img_.shape[0]:, mapped_img_.shape[1]:] = mapped_img_

        return mapped_img, (self._left_lane, self._right_lane)


def main():
    """
    Test script
    :return: True if execution was successful
    """
    from matplotlib import pyplot as plt

    test_img_path = ["/Users/aponamaryov/Desktop/test6.png",
                     "/Users/aponamaryov/Desktop/test5.png",
                     "/Users/aponamaryov/Desktop/test4.png",
                     "/Users/aponamaryov/Desktop/test1.png",
                     "/Users/aponamaryov/Desktop/test2.png",
                     "/Users/aponamaryov/Desktop/test3.png"]
    apf = APF(calibration_file_path='src/calibration.p', frames=8,
              h_threshold=(16, 24), y_threshold=(200, 256), direction_threshold=(0.0, 1/6 * np.pi),
              sobel_window=(61, 120, 100), sobel_kernel=9)

    for p in test_img_path:

        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)

        mapped_img, (left, right) = apf.detect(img, debug=True)
        plt.imshow(mapped_img)

    return True

if __name__=="__main__":
    main()

