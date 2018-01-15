# **Finding Lane Lines on the Road** 

## Writeup Template


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Image processing pipeline

My pipeline consisted of 6 steps. 

1. Keep only yellow and white pixels from the original image :
	I made the assumption that lines were either mostly white or yellow. Therefore I created a dedicated function to filter out all pixels that are not either white or yellow. To achieve that, I used the function cv2.inRange with [175,175,175] and [255,255,255] as lower and higher threshold for the white; and [170,170,0] and [255,255,150] as thresholds for the yellow. The cv2.inRange function is applied to the original image for the two colors separately, and the results are combined using the cv2.bitwise_or function. The result is a black-and white image.
2. Detect edges using the canny function
	The parameter of the canny function where 80 for the lower threshold and 200 for the high threshold, which seemed to work well on all test images. No gaussian smoothing was applied at this stage since the canny edge detection function applies it by default.
3. Restrict picture to region of interest
	A polygonal region was used for this. Manual tuning of the parameters was used to obtain satisfactory results on all test images. 
4. Detect straight line segments using a hough transform
	The grid in Hough space used a spacing of rho=2 and theta =1 degree. The threshold was set to 10, and the minimum line length to 8 pixels. The maximum line gap was set to 5. Again, these value were manually tuned to obtain good results on the test images.
5. Draw straight lines on each side of the lane
	From the line segments detected by the Hough transform, a single line was computed for each side of the lane. This was achieved as follows :
		a. Separate segments from the left line from segment from the right line. This is based on the sign of the slope of the segment. Furthermore, segments that do not have a minimum slope of $\pm 0.3$ are purely ignored. This remove the spurious horizontal segments that may have been detected.
		b. For each side, compute the average slope of all the segments, as well as the centroid of the cloud of points made by the segments endpoints. The average slope is the slope of the line we are trying to draw, while the coordinates of the centroid is one point, we want the line to pass through.
		c. Knowing this information, we can compute the equation of the line. 
		d. To determine its endpoints, we decide that the lines should start at the bottom of the image and extend up to the coordinates $y=340$ (arbitrary value). To find the $x$ coordinates, we can use the equation of the line.

6. Apply gaussian filter to the lines.
Just for visual  comfort, a gaussian filter was applied on the straight line to make them look more straight.

### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming is that many values are hard coded and probably works well only for the test images and videos that are available. For instance, if the images has a different size, the parameters defining the region of interest are no longer applicable. 

Another shortcoming could be the color filter applied at the beginning. If lines happen to be any other color than white or yellow (in a public work area for instance),  or if the external lighting would make the line less visible, the algorithm would fail to detect them. 

Furthermore, in the video rendering, the straight lines looks a bit unstable. This is because the lines computed in two successive frames are independent from each other. Hence, the slope of the lines changes a bit from a frame to the next one.

Finally, since we end up drawing one single straight line on each side of the car, it only work if the lane does not turn too much. 


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to find tranform parameters that are universal based on the global dimension of an image.

Another potential improvement could be to make curvy lines on each side instead of straight lines of the lane to accomodate for turns. This could be achieved using some kind of local regression or a piecewise linear line with a slope changing with the $y$ coordinate. 
