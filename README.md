# **Advanced Lane Finding**
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
---
## Files and data locations

The file layout for this project is extremely simple.

All code is located in **"./Advanced_Lanes.ipynb"**

All example images are location in **"./writeup_images/"**.

Output videos are located in the root directory of the project.  Naming convention and
content are covered below.

The directories **"./camera_cal"** and **"./test_images"** are provided by udacity.  The
test images directory includes a few extra images not used or referenced in this writeup.

---

**Advanced Lane Finding Project**

The goals of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup_images/undistorted_chessboard.jpg "Undistorted"
[image2]: ./writeup_images/undistorted_straight_lines1.jpg "Undistorted Road Image"
[image3]: ./writeup_images/grandient_color.jpg "Set of Gradient and Color Channel Images"
[image4]: ./writeup_images/top_view.jpg "Top View Transform"
[image5]: ./writeup_images/sliding_window.jpg "Sliding Window 2 Samples"
[image6]: ./writeup_images/cont_sliding_window.jpg "Already fit 2 Samples"
[image7]: ./writeup_images/lane_marking_debug.jpg "Lane Marking with debug data"
[image8]: ./writeup_images/motorcycle.jpg "Motorcycle Lane Occlusion"
[image9]: ./writeup_images/no_data_to_use.jpg "No Lane Data"


[video1]: ./project_video.mp4 "Video"

## [Rubric Points](https://review.udacity.com/#!/rubrics/571/view)

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. You're reading it!

### Camera Calibration

#### 1.  I used the code from our the lessons associated with this project:

- [Finding Corners](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/40ec78ee-fb7c-4b53-94a8-028c5c60b858/concepts/bf149677-e05e-4813-a6ea-5fe76021516a)
- [Calibrating Your Camera](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/40ec78ee-fb7c-4b53-94a8-028c5c60b858/concepts/a30f45cb-c1c0-482c-8e78-a26604841ec0)
- [Correcting for Distortion](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/40ec78ee-fb7c-4b53-94a8-028c5c60b858/concepts/5415176a-d615-49af-8535-53a385768a23)

I added a minimal Camera object to keep track of the needed data.

The code for this step is contained in the 2nd and 3rd code cells of the IPython notebook located in "./Advanced_Lines.ipynb"

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![distorted and undistorted image][image1]

### Pipeline (single images)

#### 1. Distortion-Corrected Image.

See above description for camera distortion.  Example of a test image with distortion correction:
![alt text][image2]

#### 2. Gradients and Color Transforms

This function extract_lane_lines() currently uses the everthing including the kitchen sink approach. It uses:
* R channel Threshold
* S Channel Threshold
* H Channel Threshold
* Yellow and White mask Threshold(s)
* X Gradient Threshold

It uses a voting method to determine a second threshold. The X gradient gets 2 votes the others get 1.
When analyzed on the challenge video, the yellow/white masks primary contribution seems to be noise. It should probably be dropped.

The R channel is really about the only required data for the basic video, but has significant problems on the challenge video.
This is the function that would likely benefit the most from significant work on the parameters of each channel and the addition of some type of adaption to changing conditions that change the parameters of the search for features.

![alt text][image3]

#### 3. Perspective Transform

I use 2 functions to perform the perspective tranform.  top_view_setup() computes the transformation
matrixes.  top_view_transform() performs the transformation on a supplied image.

This code is in the 12th code cell of the IPython notebook "./Advanced_Lines.ipynb".  It is labeled with the section header
**'Image Transform'**.

I started with an initial guestimate of the points of the transform trapezoid on one of the 
straight lines test images.  This was somewhat inaccurate and orginally gave a horizon that was
far too long for the challenge video.

I reduced the horizon.  I used test plotted lines to debug the transform as I had the coordinates
initially reveresed.  I also used them to confirm the transformation when finally corrected.

I also used the tranformation to "pull in" the X axis to make more of the lines from the challenge
video be available in the transformed image.

I settled on the following points for my src and dst coordinates:

```python
transform_left_intercept = 230
transform_right_intercept = 1070

    bl = (transform_left_intercept, 720-28)
    br = (transform_right_intercept, 720-28)
    #tl = (598, 720-275) # Original trapezoid from eyeballing image
    #tr = (683, 720-275)
    tl = (550, 720-250) # Adjustment for straightness, also shortens horizon for challenge video
    tr = (730, 720-250)

    # Manually tweaked trapezoid adjusted to make straight lines look straight on the top down view.
    
    # Destination points
    # Also adjusted to bring in the lower lane offsets to get more of the lane in view on curved roads.
    nbl = (bl[0]+100, bl[1])  
    nbr = (br[0]-100, br[1])
    ntl = (bl[0]+100, 0)
    ntr = (br[0]-100, 0)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 550, 470      | 330, 0        | 
| 230, 692      | 330, 692      |
| 1070, 692     | 970, 692      |
| 730, 470      | 970, 0        |


![alt text][image4]

#### 4. Lane Line Identification

This code is location in section **"Sliding Window Search"** of the IPython notebook.

This code was taken almost entirely from the projection lesson 33.  It uses a histogram
to identify the 2 largest peaks, one of left and one on right as the starting point
for the sliding windown search.

It then walks through defining a set of windows surounding the initial point and 
subsequent window centers.  Window recentered/placement is dependent on the margin and
minimum pixel parameters that were left unchanged from the sample code.

After the windows are defined, the pixels that are assumed to be lane line pixels are
identified.

I fit a second order polynomial to these points with np.polyfit().

![alt text][image5]

I also implemented a function cont_sliding_windows(), it essentially uses the fit provided,
finds the points within a margin of that fit and recomputes the fit based on the found
points.  This is also based on the code from the class lessons in 33 and 34.

Below is an exmaple of the 2 functions in this section:

![alt text][image6]


#### 5. Radius of curvature

This code is in the section "Radius of curvature" in the IPython notebook.

By the time I implemented this, somehow I forgot about the project lesson that covered the radius of curvature. Googling I found the same formula. I really only ended up with a difference in the distance per y pixel value which is explained in the code comments:

```python
# Top view shows about 1.5 cycles of dashed lines at 40ft per dash plus space
# Note this is sensitive to the prespective transform.  Really should be based on it.
```

#### 6. Lane Marking

My lane marking function is draw_final() in the section "Draw Final" of the IPython notebook.

This function cv2.fillPoly() to display the lane points and cv2.warpPerspective() to transform 
the lane data back to the original image perspective.

This function also annotates the image with the radius of curvature information and the
distance from center metric.

This example image also contains debug data and stats about the fit and an still warped top view image
of the lane fit.

![alt text][image7]

---

### Pipeline (video)

#### 1. Video Output

My code performs quite well on the project video, there is some small waviness near the lane horizon
on a few frames.  It does loose the fit on one frame on the cont_sliding_windows() and drops back to
the function initial_sliding_windows() to reaquire the lane lines.

The challenge video has significant problems.  I'll mention some in the discussion section.

Here's a [link to my video result.](./project_video_output.mp4)

Here's a [link to my video result with partial debug data.](./project_video_debug_output.mp4)

Here's a [link to my video result from the challenge video.](./harder_challenge_video_output.mp4)

---

### Discussion

#### 1. Difficulties

I had trouble finding a reasonable method to combine information from different image components
such as the S Channel and X gradient threshold and others into a single value.  I was not happy
with logical combination(and, or, masks...).  I ended up with a scoring/voting method that also
has some problems.  See improvements below on this.

I had significant difficulties in figuring out how to combine multiple video streams and text 
content on the final output and debug output videos.  This was mostly the result of not knowing the
capabilities of the libraries.

Visualization of some data was difficult.  Combining images and graphs, as in one of my histogram 
images, was suprisingly difflcult and surprisingly hard to find related examples via google.


#### 2. Where will likely fail?

In general I think the weakest part of my is the lane data extraction from the raw image.  One example
is this frame where there is virtually no data for the rest of the code to use:

![alt text][image9]

My code will fail when either shadows or in this case a motor cycle occludes parts of the line.
The 2nd sliding window on the left in this image also appears to center incorrectly or at least
not optimally.  Possible improvement is to center the window based on the direction of the line
if reasonable to capture.

On the right lane, the lack of the region of interest and noise on the road side starts the
initial sliding window in the wrong place. 

![alt text][image8]

#### 3.  Improvements

First my code needs a region of interest.  It is not required for the project video but the challenge 
video needs it.

I feel like this current code needs some type of adaptive filtering for the initial image processing.
I don't think it can be done frame by frame since many of the problems in the challenge video present
multiple lighting conditions within the same frame.  Perhaps using multiple channel and gradient combinations
and switching between them on a per window basis in the sliding window code could be helpful.

An alternate scoring method might be helpful as well, perhaps based on a whole image or region statistics
on saturation or other feature.

I would like to experiment with shorter and narrower sliding windows, although the challendge video
clearly has curves that don't fit within the current margin.

I do not have a robust method for detecting the failure of sliding window.  Perhaps I could base this on
the prior fit.
