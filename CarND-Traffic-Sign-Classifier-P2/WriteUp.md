# Traffic Sign Recognition

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Figures/histogram.png "Visualization"
[image2]: ./Figures/preprocessing.png "Preprocessing"
[image3]: ./Figures/training.png "Training"
[image4]: ./Figures/newsigns.png "Traffic Signs"

[image11]: ./Figures/prob1.png "Top 5 Predictions for sign 1"
[image12]: ./Figures/prob2.png "Top 5 Predictions for sign 2"
[image13]: ./Figures/prob3.png "Top 5 Predictions for sign 3"
[image14]: ./Figures/prob4.png "Top 5 Predictions for sign 4"
[image15]: ./Figures/prob5.png "Top 5 Predictions for sign 5"
[image16]: ./Figures/prob6.png "Top 5 Predictions for sign 6"
[image17]: ./Figures/prob7.png "Top 5 Predictions for sign 7"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

*1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.*

You're reading it! and here is a link to my [project code](https://github.com/mkinet/CarND-Traffic-Sign-Classifier-P2/)


###Data Set Summary & Exploration

*1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.*


I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

*2. Include an exploratory visualization of the dataset.*

Here is an exploratory visualization of the data set. It is a bar chart showing the proportion of each traffic sign in the training, validation and test set.

![alt text][image1]

We can observe that in general, the training and test sets are more or less identically distributed, but the validation set is slightly different.

###Design and Test a Model Architecture

*1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)*

As a first step, I decided to convert the images to grayscale because I wanted to keep the complexity to a reasonable level, and colors add complexity ot the process. Besides, since we are standardizing each image, and traffic signs are shown in front of background of very different colors, the color information of the traffic signs will be altered in any case.

As a last step, I normalized the image data because convolutional nets work better with data data are centered on 0 and with similar scales.

Here is an example of a traffic sign image before and after the grayscaling and normalization operations.

![alt text][image2]


*2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.*

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| dropout				| Keep prob =0.5 at train time. 1.0 at test time|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 24x24x32 	|
| RELU					|												|
| dropout				| Keep prob =0.5 at train time. 1.0 at test time|
| Max pooling	      	| 2x2 stride, outputs 12x12x64 				    |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 8x8x64 	|
| RELU					|												|
| dropout				| Keep prob =0.5 at train time. 1.0 at test time|
| Max pooling	      	| 2x2 stride,  outputs 4x4x64 			     	|
| Fully connected		| 256 hidden units                              |      									|
| Softmax				| 43 hidden units        						|			|
|						|												|
 


*3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.*

The optimization is achieved using AdamOptimizer with an initial learning rate of 5e-3.
In an attempt to reduce overfitting, an L2 regularization term with regularization parameter equal to 1e-3 was added. 
The network is trained on 15 epochs (i.e. passes over the entire training set), by batches of 64 training examples. After each epoch, we compute the accuracy of the classification over the training and validation sets. 
Accuracy on the test set, is computed at the end of the process. 

The training was made on an AWS instance with GPU, which was pretty fast and allowed several repetition for tuning the parameters. 

*4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.*

My final model results were:
* mini-batch set accuracy of 1.0
* validation set accuracy of 0.97
* test set accuracy of 0.96

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

The first architecture that was chosen was LeNet as exposed in the LeNet lab. this architecture was chosen because it was readily implemented and hence provided a rapidly available solution. Then I added one convolutional layer and removed on fully connected layer to reduce the number of weights. Learning rate and batch size were tune manually. I noticed that there was most likely too much overfitting as prediction on the training set is almost immediately perfect. I could reduce overfitting by adding dropout and a regularization term to the loss function, but the performance on training set is still much better than onthe validation set.

The figure below shows the evolution of the accuracy with the number of training epoch for the validation set and one mini-batch.

![alt text][image3]

###Test a Model on New Images

*1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.*

Here are seven German traffic signs that I found on the web:

![alt text][image4] 

The first five images should not be too difficult to classify. The sixth image might be difficult because the size of the sign is quite small. The last one might be difficult because of the camera position which makes the sign looks skewed.

*2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).*

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|Stop                   | Stop                                          |
|Speed limit (50km/h)   | Speed limit (50km/h)                          |
|No entry               | No entry                                      |
|Right-of-way at the next intersection|Right-of-way at the next intersection|
|Wild animals crossing  | Wild animals crossing                         |
|Double curve           | No entry                                      |
|Speed limit (80km/h)   | Keep Left                                     |

The model was able to correctly guess 5 of the 7 traffic signs, which gives an accuracy of 71%. 

*3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)*

The code for making predictions on my final model is located in the Ipython notebook.

The figures below shows for each image the top 5 softmax probabilities. 

![alt text][image11] 
![alt text][image12] ![alt text][image13] 
![alt text][image14] ![alt text][image15] ![alt text][image16]
![alt text][image17]

We can see that the five first images were correctly predicted and that, for those five images, the network is pretty confident in its prediction (very high probability of classification).

The prediction on the last two images are completely wrong : the correct class is not even in the top 5 predictions. This indicate that the network has a lot of problem with these images that are not exactly similar to what the training set contains. This problem could be improved by augmenting the training set and adding copies of training examples with slight distortion such as scaling, translation, skewing, rotation, ... This is also another indication that the network might be overfitting and therefore has troubles generalizing.


