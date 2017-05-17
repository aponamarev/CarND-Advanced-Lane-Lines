**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/calibrated_test_images.png "Road Transformed"
[image3]: ./output_images/binary_image.png "Binary Example"
[image4]: ./output_images/warped.png "Warp Example"
[image5]: ./output_images/curve.png "Curve"
[image6]: ./output_images/mapped_img.png "Output"
[video1]: ./project_video.mp4 "Video"

---

# README

This README provides a description of the main steps of CarND-Advanced-Lane-Lines project.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The implementation of camera calibration step is contained in src/CameraCalibration.py file.

To avoid recalculating calibration statistics at every run, I start by checking whether it is possible to use a premade calibration file (src/CameraCalibration.py - lines 21-27). If the calibration file doesn't exist, I will use a provided path to save the calibration stats. Next, I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image (src/CameraCalibration.py - lines 33-34). Thus, `objp` is just a replicated array of coordinates, and `objps` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgps` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection (src/CameraCalibration.py - lines 54-55).  

I then used the output `objps` and `imgps` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using `cv2.getOptimalNewCameraMatrix` and `cv2.undistort()` functions and obtained this result: 

![alt text][image1]

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to images like these:
![alt text][image2]

Correction of image distortion (created by the camera) is done in 2 steps: a) calculate cameral matrix and distortion coefficients and b) undistort an image using statistics calculated in the step a.
The first step was implemented in a utility class src/CameraCallibration.py. The script in aforementioned file processed all provided examples (chessboard images) and stored resulting stats in src/calibration.p. Therefore, in advanced_paths_finding.ipynb I focused on the step b).
Step b) consists of creating an CameraCalibration object and using this object to undistort an image. The code for this step provided in 1 and 2 code cells of the book (advanced_paths_finding.ipynb).

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in code cells 5 and 6 in `advanced_paths_finding.ipynb`).  Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 70 through 76 in the file `src/utils.py` and in the 7th code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 60), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 700, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In order to detect lane pixels I applied a sliding window method. Sliding window method divides the images into n windows. The maximum activations found in each window considered to be the centers of the lane or centorid. Therefore, binary active pixels sournding this centroid will be considered belonging to the lane.

The method of finding centroids described in map_window method of src/utils.py

The results are presented below:

![alt text][image5]

The code presented in code cells 11 and 12 of `advanced_paths_finding.ipynb`

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To calculate the curvature I applied the formula below:

R(f(y)) = (1+f(y)'^2)^(2/3) / |f(y)''|, where:
* f(y) = Ay^2 + By + C
* f(y)' = 2Ay + B f(y)'' = 2A

Therefore: R(f(y)) = (1+(2Ay+B)^2)^(2/3) / |2A|

Left radius in meters: 546.2
Right radius in meters: 564.5

The code presented in code cells 12 of `advanced_paths_finding.ipynb`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
