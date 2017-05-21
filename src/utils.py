#!/usr/bin/env python
"""CameraCalibration.py is class that provides camera calibration tools."""

__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"

import numpy as np
from matplotlib import pyplot as plt
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
    mask = np.zeros_like(absgraddir, dtype=np.uint8)
    mask[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return mask


def warper(img, src=None, dst=None, top_c_shift = 45, top_v_shift = 90):
    """
    Warps an image input based on provided rectangle coordinates for the source image (src) and 
    the destination image (dst)
    :param img: input image
    :param src: rectangle coordinates for the source image
    :param dst: rectungle coordinates for the destination image
    :return: warped image
    """
    # Set img_size and transformation points
    if len(img.shape)==3:
        img_size = img.shape[:-1][::-1]
    elif len(img.shape)==2:
        img_size = img.shape[::-1]
    else:
        assert False, "Provided image has incorrect shape {}. Image expected to have 2 or 3 dimentions".format(img.shape)

    src = src if src is not None else np.float32([[(img_size[0] / 2) - top_c_shift, img_size[1] / 2 + top_v_shift],
                                                 [((img_size[0] / 6) - 10), img_size[1]],
                                                 [(img_size[0] * 5 / 6) + 60, img_size[1]],
                                                 [(img_size[0] / 2 + top_c_shift), img_size[1] / 2 + top_v_shift]])
    dst = dst if dst is not None else np.float32([[(img_size[0] / 4), 0],
                                                 [(img_size[0] / 4), img_size[1]],
                                                 [(img_size[0] * 3 / 4), img_size[1]],
                                                 [(img_size[0] * 3 / 4), 0]])
    # Create a transformation matrix and warp the image
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, img_size), (src, dst)


def visualize_planes(img, plane):
    for pt1, pt2 in zip(plane, np.roll(plane,1, axis=0)):
        img = cv2.line(img, (pt1[0],pt1[1]), (pt2[0], pt2[1]), (255,0,0), thickness=3)
    return img


def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


def find_window_centroids(image, window_width, window_height, margin, limit=10):
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = [i for i in range(1,window_width//2+2)] + [i for i in range(window_width//2,0,-1)]  # Create our window template that we will use for convolutions
    offset = window_width/2

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
    # Calculate the center (x coordinate) for left lane
    l_center = np.argmax(np.convolve(window, l_sum)) - offset
    #l_center = np.argmax(np.convolve(window, l_sum))
    r_sum = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - offset + int(image.shape[1] / 2)
    #r_center = np.argmax(np.convolve(window, r_sum)) + int(image.shape[1] / 2)

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(image.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            image[int(image.shape[0] - (level + 1) * window_height):int(image.shape[0] - level * window_height), :],
            axis=0)
        # Find the best left centroid by using past left center as a reference
        # Use _window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        conv_signal = np.convolve(window, image_layer)

        l_min_index = int(max(l_center+offset - margin, 0))
        l_max_index = int(min(l_center+offset + margin, image.shape[1]))
        l_conv = conv_signal[l_min_index:l_max_index]
        if l_conv.max()>limit:
            l_center = np.argmax(l_conv) + l_min_index - offset
            #l_center = np.argmax(l_conv) + l_min_index
        else:
            l_center = None
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset - margin, 0))
        r_max_index = int(min(r_center+offset + margin, image.shape[1]))
        r_conv = conv_signal[r_min_index:r_max_index]
        if r_conv.max() > limit:
            r_center = np.argmax(r_conv) + r_min_index - offset
            #r_center = np.argmax(r_conv) + r_min_index
        else:
            r_center = None
        if (l_center is None) & (r_center is None):
            l_center, r_center = window_centroids[-1]
        else:
            l_center = r_center - (window_centroids[-1][1]-window_centroids[-1][0]) if l_center is None else l_center
            r_center = l_center + (window_centroids[-1][1] - window_centroids[-1][0]) if r_center is None else r_center
        # Add what we found for that layer
        window_centroids.append((l_center, r_center))

    return np.int32(window_centroids)

def map_window(warped_img, window_centroids, window_width, window_height):

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped_img)
        r_points = np.zeros_like(warped_img)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, warped_img, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, warped_img, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        l_points = np.where(((warped_img>0) & (l_points>0)).astype(l_points.dtype))
        r_points = np.where(((warped_img > 0) & (r_points > 0)).astype(r_points.dtype))


    # If no window centers found, just display orginal road image
    else:
        l_points = []
        r_points = []

    return l_points, r_points

def fit_polynomial(x, y_size, x_scaller=1.0, y_scaller=1.0, degree=2):

    # Define y space
    ploty = np.linspace(0, y_size-1, len(x))
    ploty = ploty[::-1]

    # Fit polynomial
    return np.polyfit(ploty*y_scaller, x*x_scaller, degree)

def plot_2nd_degree_polynomial(coefficients, y_size):

    # Define y space
    ploty = np.linspace(0, y_size - 1, y_size)
    ploty = ploty[::-1]

    # Build points
    x = coefficients[0]*ploty**2 + coefficients[1]*ploty + coefficients[2]

    return x, ploty

def curvature(centroids, y_length, x_conv = 3.7 / 700, y_conv = 30 / 720):

    # Fit polinomial
    left_fit = fit_polynomial(centroids[:, 0], y_length)


def visualize_lanes(warped, ploty, left_fitx, right_fitx):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    return color_warp


def main():

    from CameraCalibration import CameraCalibration as CC
    from matplotlib import pyplot as plt

    calibration_file_path = 'calibration.p'

    test_img_path = "../test_images/test3.jpg"

    cc = CC(calibration_file_path)

    img = cv2.cvtColor(cv2.imread(test_img_path), cv2.COLOR_BGR2RGB)
    img = cc.undistort(img)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    binary_img = abs_sobel_thresh(gray, thresh=(20,100))
    #binary_img = mag_thresh(gray, thresh=(20,100))

    top_c_shift = 40
    top_v_shift = 85

    warped_img, (src, dst) = warper(img, top_c_shift=top_c_shift, top_v_shift=top_v_shift)
    img = visualize_planes(img, src)
    warped_img = visualize_planes(warped_img, dst)

    """
    fig, (sub_plot1, sub_plot2) = plt.subplots(1,2, figsize=[20,6])
    sub_plot1.set_title("original")
    sub_plot1.imshow(img)
    sub_plot2.set_title("warped")
    sub_plot2.imshow(warped_img)"""

    binary_warped_img, (src, dst) = warper(binary_img, top_c_shift = top_c_shift, top_v_shift = top_v_shift)
    """
    fig, (sub_plot1, sub_plot2) = plt.subplots(1, 2, figsize=[20, 6])
    sub_plot1.set_title("original")
    sub_plot1.imshow(binary_img, cmap="gray")
    sub_plot2.set_title("warped")
    sub_plot2.imshow(binary_warped_img, cmap="gray")"""


    mapped_binary, centroids = map_window(binary_warped_img, 50, 80, 100)


    # Display the final results
    """
    fig, (sub_plot1, sub_plot2) = plt.subplots(1, 2, figsize=[20, 6])
    sub_plot1.set_title("warped")
    sub_plot1.imshow(binary_warped_img, cmap="gray")
    sub_plot2.set_title("maped")
    sub_plot2.imshow(mapped_binary)"""


    # Fit polinomial
    left_fit = fit_polynomial(centroids[:, 0], binary_warped_img.shape[0])
    right_fit = fit_polynomial(centroids[:, 1], binary_warped_img.shape[0])
    print("Left:", left_fit)
    print("Right:", right_fit)

    # Plot curves
    left_x, left_y = plot_2nd_degree_polynomial(left_fit, binary_warped_img.shape[0])
    right_x, right_y = plot_2nd_degree_polynomial(right_fit, binary_warped_img.shape[0])

    """
    plt.title("plotted curves")
    plt.plot(left_x, left_y, 'o', color='red', linewidth=3)
    plt.plot(right_x, right_y, 'o', color='blue', linewidth=3)
    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    plt.gca().invert_yaxis()
    plt.show()"""

    # Calculate radius of curvature
    # R(f(y)) = (1+f(y)'^2)^(2/3) / |f(y)''|, where:
    # f(y) = A*y^2 + B*y + C
    # f(y)' = 2Ay + B
    # f(y)'' = 2A
    # Therefore:
    # R(f(y)) = (1+(2Ay+B)^2)^(2/3) / |2A|

    R = lambda M, y: (1+(2*M[0]*y+M[1])**2)**(2/3) / np.abs(2*M[0])

    x_scaller = 3.7 / 580
    y_scaller = 8.5*3 / 720

    left_fit = fit_polynomial(centroids[:, 0], binary_warped_img.shape[0], x_scaller=x_scaller, y_scaller=y_scaller)
    right_fit = fit_polynomial(centroids[:, 1], binary_warped_img.shape[0], x_scaller=x_scaller, y_scaller=y_scaller)

    left_curve_rad = R(left_fit, left_y).mean()
    right_curve_rad = R(right_fit, right_y).mean()
    car_position = binary_warped_img.shape[1]/2-np.mean(centroids[0])

    print("Left radius in meters: {:.1f}".format(left_curve_rad))
    print("Right radius in meters: {:.1f}".format(right_curve_rad))

    binary_filled = visualize_lanes(binary_warped_img, left_y, left_x, right_x)
    unwarp_binary, (src, dst) = warper(binary_filled, src=dst, dst=src)
    mapped_img = cv2.addWeighted(img, 1, unwarp_binary, 0.3, 0)
    cv2.putText(mapped_img, "Radius of curvature: {}m".format((left_curve_rad+right_curve_rad)//2),
                (5, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), thickness=1)
    cv2.putText(mapped_img, "Vehicle is {:.1f}m {} of center".format(np.abs(car_position*x_scaller), "left" if car_position<0\
        else "right"),
                (int(mapped_img.shape[1]/2), 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), thickness=1)

    plt.title("mapped_img")
    plt.imshow(mapped_img)
    plt.show()


    return True

if __name__=="__main__":
    main()

