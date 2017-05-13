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

    src = src or np.float32([[(img_size[0] / 2) - top_c_shift, img_size[1] / 2 + top_v_shift],
                             [((img_size[0] / 6) - 10), img_size[1]],
                             [(img_size[0] * 5 / 6) + 60, img_size[1]],
                             [(img_size[0] / 2 + top_c_shift), img_size[1] / 2 + top_v_shift]])
    dst = dst or np.float32([[(img_size[0] / 4), 0],
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


def find_window_centroids(image, window_width, window_height, margin):
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
    # Calculate the center (x coordinate) for left lane
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    r_sum = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(image.shape[1] / 2)

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    offset = window_width / 2

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(image.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            image[int(image.shape[0] - (level + 1) * window_height):int(image.shape[0] - level * window_height), :],
            axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window

        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        # Add what we found for that layer
        window_centroids.append((l_center, r_center))

    return window_centroids

def map_window(binary_warped_img, window_width=50, window_height=120, search_margin=100):

    window_centroids = find_window_centroids(binary_warped_img, window_width, window_height, search_margin)

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(binary_warped_img)
        r_points = np.zeros_like(binary_warped_img)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, binary_warped_img, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, binary_warped_img, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
        warpage = np.array(cv2.merge((binary_warped_img, binary_warped_img, binary_warped_img)),
                           np.uint8)  # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5,
                                 0.0)  # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((binary_warped_img, binary_warped_img, binary_warped_img)), np.uint8)

    return output, window_centroids


def main():

    from CameraCalibration import CameraCalibration as CC
    from matplotlib import pyplot as plt

    calibration_file_path = 'calibration.p'

    test_img_path = "../test_images/straight_lines1.jpg"

    cc = CC(calibration_file_path)

    img = cv2.cvtColor(cv2.imread(test_img_path), cv2.COLOR_BGR2RGB)
    #img = cc.undistort(img)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    binary_img = abs_sobel_thresh(gray, thresh=(20,100))

    top_c_shift = 29
    top_v_shift = 80

    warped_img, (src, dst) = warper(img, top_c_shift = top_c_shift, top_v_shift = top_v_shift)
    img = visualize_planes(img, src)
    warped_img = visualize_planes(warped_img, dst)

    fig, (sub_plot1, sub_plot2) = plt.subplots(1,2, figsize=[20,6])
    sub_plot1.set_title("original")
    sub_plot1.imshow(img)
    sub_plot2.set_title("warped")
    sub_plot2.imshow(warped_img)

    binary_warped_img, (src, dst) = warper(binary_img, top_c_shift = top_c_shift, top_v_shift = top_v_shift)

    fig, (sub_plot1, sub_plot2) = plt.subplots(1, 2, figsize=[20, 6])
    sub_plot1.set_title("original")
    sub_plot1.imshow(binary_img, cmap="gray")
    sub_plot2.set_title("warped")
    sub_plot2.imshow(binary_warped_img, cmap="gray")


    mapped_binary, centroids = map_window(binary_warped_img, 50, 120, 100)


    # Display the final results
    fig, (sub_plot1, sub_plot2) = plt.subplots(1, 2, figsize=[20, 6])
    sub_plot1.set_title("warped")
    sub_plot1.imshow(binary_warped_img, cmap="gray")
    sub_plot2.set_title("maped")
    sub_plot2.imshow(mapped_binary)

    return True

if __name__=="__main__":
    main()

