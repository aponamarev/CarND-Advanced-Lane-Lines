#!/usr/bin/env python
"""Lane.py is class that defines lane tracking pipeline."""

__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"

import numpy as np

class Lane(object):
    """ Class designed to track lanes"""
    def __init__(self, name, frames=8, bounds_diffs = np.array([1e-4, 1.0, 100.0], dtype='float')):
        self._frames = frames
        self.__name__=name
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self._recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self._current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self._diffs = np.array([0, 0, 0], dtype='float')
        # bounds for the difference in fit coefficients
        self._bounds_diffs = bounds_diffs
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    @property
    def current_fit(self):
        return self._current_fit
    @current_fit.setter
    def current_fit(self, value):
        # store value
        self._current_fit=value
        # add value to the buffer
        if len(self._recent_xfitted)>=self._frames:
            self._recent_xfitted.pop(0)
        self._recent_xfitted.append(value)
        # calculate difference in fit coefficients between last and new fits
        if self.best_fit is not None:
            self._diffs = np.abs(self.best_fit - value)
        # assign best fit
        # assign current fit if it is in expected bounds
        if self._diffs[0]<self._bounds_diffs[0] and self._diffs[1]<self._bounds_diffs[1]:
            self.best_fit=value
        else:
            self.best_fit = np.array(self._recent_xfitted).mean(axis=0)




