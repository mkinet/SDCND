# Advanced Lane detection project

---


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

[image1]: ./Figures/Undistorted1.png "Undistorted 1"
[image2]: ./Figures/Undistorted2.png "Undistorted 2"
[image3]: ./Figures/Thresholds1.png "Binary Example"
[image4]: ./Figures/Thresholds2.png "Binary Example"
[image5]: ./Figures/Perspective1.png "Perspective Transform"
[image6]: ./Figures/Perspective2.png "Perspective Transform"
[image7]: ./Figures/LineFitting.png "LineFitting"
[image8]: ./Figures/FullPipeline1.png "FullPipeline"
[image9]: ./Figures/FullPipeline2.png "FullPipeline"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Camera Calibration

The code for this step is contained in the cells 2 to 4 of Jupyter notebook  named `Advanced_Lane_lines.ipynb`.  

The goal of camera calibration is to compute the distortion of the image induced when the camera transforms a 3D object into a 2D image. The transformation is in general imperfect and distortion will change the shape and size of the objects. Once the distortion parameters of the camera are know, when can use them to remove the effects of distortion and recover correct information on the object. 

To compute the distortion, we have at our disposal several chessboards images (i.e. whose correct shapes are known). For each of these image we need to : 
* convert the image to grayscale,  
* use the function `cv2.findChessboardCorners` to automatically detect the corners of the chessboards
* use the `cv2.calibrateCamera` function to compute the distortion parameters of the camera. This is done by computing a mapping between the corners found at previous step and their position without distortion.
* apply the `cv2.undistort` function with parameters computed by the previous function to correct distortion in the following image.

After converting to grayscale, I prepare "object points", which will be the (x, y, z) coordinates of the chessboard corners in the physical world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  For each chessboard image, the (x,y) pixel positions of the successfully found corners will be stored, along with there corresponding 'object point ' location'.   

The 'object points' and corresponding 'image points' are then used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to a calibration image image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]


### Pipeline (single images)

##### Example of a distortion-corrected image.

The following image shows the distortion correction applied on one of the test image. We can see on the right side how the position of the white car is modified due to distortion correction

![alt text][image2]

##### Gradients and color thresholds 

I used a combination of gradient and colors thresholds to generate a binary image. These operations are coded and tested at cells 6 to 11 of jupyter notebook.

The gradient threshold operation consist in applying sobel filters and keep only the pixels that meet one of the two following conditions :
1. scaled gradient in x direction AND gradient in the x direction are between 25 and 255

OR 

2. scaled gradient magnitude is larger than 70 and direction of gradient is between 0.7 and 1.3 radians.

Furthermore, a binary trapezoidal mask is applied on the gradient thresholded image to get rid of spurious lines that may appear outside of the lane (e.g caused by shade of the left wall)

The color gradient consist in restricting the pixels to those that meet the following conditions :
1. S value of HLS coding is larger than 100

AND 

2. V value of HSV coding is larger than 50.

The final binary image is made of the pixels that where not discarded by either the gradient filter or the gradient filter. The result of the thresholding process is shown in the next two image, where the original image is also shown for comparison.

![alt text][image3]
![alt text][image4]

##### 3. Perspective transform

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

Perspective transform is studied and implemented in cells 13 to 15 of jupyter notebook. The perspective transform parameters are computed by mapping a trapezoidal shape on a rectangular shape. The trapezoidal shape was defined to cover more or less the lane up to a certain ahead distance. 

The following value are used as source and destination points for the perspective transform. 

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 598, 446      | 320, 0        | 
| 240, 673      | 320, 720      |
| 1040, 673     | 960, 720      |
| 682, 446      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. This is shown on the next two images.

![alt text][image5]
![alt text][image6]

##### Lane detection
The lane detection is coded and tested in cells 16 - 17 of the notebook. It works as follows :
- First we find the most likely position of the lane lines at the bottom of the image. This is done by summing the pixels rowwise on the lower half image and assuming that the line is located at the maxium of this histogram
- Then, starting from this point, we slide a window upward and keep only the pixels that fall within that window.
- the position of the window is shifted left or right if the center of the pixel cloud within that window changes. 
- the slided windows allow us to retain only those pixels that belongs to a mask. 
- A polynomial fit is then applied to compute the equation of the line. 

We do this independently for the left and right lines. The image below shows the sliding windows (in green), the detected lane pixels (in red and blue) and the fitted line.

The result of this operation is shown on the figure below. 

![alt text][image7]

##### Curvature radius and distance from center
The curvature radius is computed at cell 18 of the notebook. We compute the radius in meters from the fit in pixel space. The formula used is derived just above cell 18.

The distance of the vehicle to the center of the road is computed using the distance of the vehicle to each of the two lines. It assumes that the camera is centered on the vehicle and therefore is at the middle of the image.


##### Full pipeline

The following images shows the full pipeline applied on one of the test image. 



![alt text][image8]
![alt text][image9]

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video.mp4)

To make the line detection process more robust, we use some tracking system of the fitting parameters, the lane curvature and the distance to the centers. The lines shown on the videos, and the values correspond to an average of each of these parameters over the last 10 frames. 



### Discussion

The process works reasonably on the most simple video. The calibration seems to be done correctly. The filter do a relatively good job at retaining only the pixels of interest. It still suffers a bit from shadow effect, especially the shadow from a side wall. 
 
The perspective works ok as well, since the left and right lines seem to be fairly parallel on every test image. The line detection function works as well, although the fit might not be perfect when we have a dashed line with not too much pixel detected on one side.

The radius of curvature is of an order of magnitude that makes sense and the same goes for the distance between the lines. 
 
There are several improvements that can be mentioned though : 
 
 - The line equation for each new frame is computed by processing the whole image. It could be speeded up by searching only in the neighborhood of the previously computed lines.
 - In case there are several frames in a row where there are large shadow effects, the process would be completely destabilized
 