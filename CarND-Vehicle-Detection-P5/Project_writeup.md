## Project Writeup
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./Figures/training_image.png
[image2]: ./Figures/hog.png
[image3]: ./Figures/sliding_windows.jpg
[image4]: ./Figures/detection_1
[image5]: ./Figures/detection_2
[image6]: ./Figures/detection_3
[image7]: ./Figures/heatmap_1
[image8]: ./Figures/heatmap_2
[image9]: ./Figures/heatmap_3
[image10]: ./Figures/heatmap_4
[image11]: ./Figures/heatmap_5
[image12]: ./Figures/heatmap_6
[image13]: ./Figures/bbox_1
[image14]: ./Figures/bbox_2
[image15]: ./Figures/bbox_3
[image16]: ./Figures/bbox_4
[image17]: ./Figures/bbox_5
[image18]: ./Figures/bbox_6


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

You're reading it!

### Features creation

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

then I implemented functions to compute spatial, histogram and HOG features (see cell 5 and 6 of the jupyter notebook).

### Histogram of Oriented Gradients (HOG) 

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.
The code for HOG features can be found in cell [6] of the Jupyter notebook 'Vehicle-Detection.ipynb'. HOG features where extracted using the skimage `hog` function.

I explored different color spaces and `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).    

Here is an example using the first channel `YUV` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

To determine the best parameter set, I made a grid search and evaluated the performance of a SVC for each parameter combination (see cell 14) by 5-fold cross-validation. Since the training set is quite large and gridsearch is an expensive operations, this was restricted on a subset of 2000 training samples selected randomly using sklearn's train_test_split. Note that stratified sampling was used to keep the class balance unchanged.

Finally, the following parameters where chosen as a compromise between feature set size and performance :

- Hog_channels : 'ALL', HOG features were computed on all three channels
- color_space : 'YUV'
- orient : 11
- pix_per_cell : 16
- cells_per_block : 2

Additionnaly, raw pixel values from a resized image of size 32x32 where used as features, and histogram features with 128 bins where also added.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
The best set of features and the best parameters where used to train a SVM classifier (see cell 21). The value of the penalty parameter was determine using cross-validation. A value of C=1e-3 was found to yield the best results. The feature vectors were standardized by column so that all features were on a comparable scale. The accuracy of the classifier, as evaluated by 5-fold cross validation is 99.1%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Sliding window search is implemented at cell 22 of the jupyter notebook. The function accepts start and stop points as inputs, as well as a window size and an overlap parameter and returns the list of patches of the given size that fit into the specified region. For the scales, I first added small windows of size 64 and 80, to detect the small size vehicles that are ahead. Then, I added bigger windows of size 96, 112 and 128 for closer vehicles. I realized that having many windows helped in the detection of the vehicle, so I settled on an overlap of 75%. 
Small scales focus on far away vehicles and thus do not extend down to the bottom of the image.

The figure below shows the window that are searched in each framed.

![alt text][image3]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on six scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
![alt text][image5]
![alt text][image6]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_videos/project_video_vehicle_detection.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The detection of a 'positive patch' is made using the function `decision_function` from sklearn's SVC. This means that the output of the classifier is not purely binary but is a number that indicates the confidence of the classifier in the prediction. Using a criteria of 0.7, I was able to reduce the number of false positive.

Then, I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

In addition, to reduce the wobbling of the bounding box on one frame to the next, I recorded the heatmaps from the last 10 frames and integrated them to produce a smooting effect.

Here's an example result showing the heatmap from the test images and the bounding boxes:

### Here are the positive patches and their corresponding heatmaps:

![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]

### Here is the output of `scipy.ndimage.measurements.label()`:
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]
![alt text][image18]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Although the pipeline works ok on the project video, it has several limitations :

- The vehicles that are too much ahead are not detected, whatever the search window I use.
- The number of search windows in each frame is quite high (>1400). This is quite time consuming and makes an operation in real time impossible.
- There are still some false positive appearing when the texture of the ground changes

Several improvements are likely to eliminate these limitations :

- We could use a CNN for the detection which would work better on the small scale vehicles and improve the accuracy of the classifier, hence avoiding the need for a large number of search windows.
- In my pipeline, HOG features are computed for each search window. We could compute the HOG features only once and then subsample it to the search window.
- The tracking of a bounding box can probably be improved to eliminate any false positive.
