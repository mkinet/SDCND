#**Behavioral Cloning** 


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./figures/center.jpg "Center View"
[image2]: ./figures/original.png "Original figure "
[image3]: ./figures/flipped.png "Image after flipping"
[image4]: ./figures/training_error.png "training/validation MSE during training"


## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

###Files Submitted and Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* output_video.mp4 showing the simulation running in autonomous mode with my model

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing: 
```sh
python drive.py model.h5
```
Note that the driving speed has been set to 20mph, instead of the initial value of 9mph which seemed terribly slow.

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file it contains comments to explain how the code works. The different steps of the process have been separated in functions instead of one big script. Each function is documented. 

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed
The model is implemented in Keras. There are first two layers that performs a basic preprocessing to the feeded images :
- Normalization layer : Normalize image arrays between -0.5 and 0.5
- Image cropping layer : Remove useless background information from the images. 50 pixels are cropped at the top and 20 at the bottom.

The architecture of the network is made of two convolutional layers (each followed by max pooling, dropout and relu activation) and two dense layers. More precisely, we have the following sequence of layers :
- Convolution layer with 16 filters of size (5,5)
   - Max Pooling with a filter of size (5,5)
   - Dropout with a dropout rate of 10%
   - ReLu activation
- Convolution layer with 32 filters of size (5,5)
   - Max Pooling with a filter of size (5,5)
   - Dropout with a dropout rate of 10%
   - ReLu activation
- Dense layer with 128 hidden units
    - Dropout with a dropout rate of 20%
    - ReLu activation
- Linear layer to compute the output.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers after each convolutional layer, and after the first dense layer in order to reduce overfitting (model.py lines 198, 202, 206). The dropout rate is moderate (10% on the convolutional layers, 20% on the dense layer) .

The model was trained on 80% of the data and validated on the 20% remaining to ensure that the model was not overfitting (code line 164). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track number one.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25). Five epochs where enough to train the model.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and image flipping. 

For details about how I created the training data, see the next section. 

###Model construction

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to build the network by increasing its complexity step by step. This allowed me to develop an acceptable model on my personal computer before turning to AWS and GPU computing.

I started with a very simple architecture (just one fully connected layer), and the model was able to drive for no more than a few meters. During training, I computed the loss on both the training set and on a validation made up of 20% of the original data. To avoid overfitting, the training was limited to 2 epochs.

Then I added one convolutional layer, followed by max pooling. The training error was reduced significantly with respect to the first model, both on the training and validation set, and the car was able to drive half a lap smoothly. It crashed in the sharp turn after the bridge. The difficulties arise whenever the car leaves the track, especially if the limits of the track are not the same as everywhere else. 

The model was overfitting significantly, even with a small number of epoch, hence I modified the model by adding dropout. 

Then I added one convolutional layer, and one dense layer. The performance improved significantly. At that point the car was able to recover autonomously if I force it slightly sideways. It had still dificulties in the sharp turn. One key improvement was to realize that opencv (which I use to read the images) reads the images in BGR format, while the simulator feed them in RGB format. Fixing this issue and making sure that the network was learning on RGB images made the car able to complete the lap autonomously without leaving the road. 


####2. Final Model Architecture 

The final model architecture (model.py lines 18-24) consisted of two convolutional layers (each followed by max pooling, dropout and relu activation) and two dense layers. For more details on the final architecture see above.

This architecture is much simpler than the NVIDIA architecture described in the paper. Since that simpler model was able to achieve the (much simpler) goal we had at hand (i.e. drive the simulator instead of a real car) I didn't feel the need to build a much more complicated system. 

In the end, the network is able to drive the car at a speed of 20mph, without leaving the track and is also able to recover if the car is pushed to the sides of the track. It fails however, if the car is placed out of the track, as the training set does not include such harsh recovery events. It still works when the PI speed is set to 30mph but the driving is much less stable.

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one and a half lap on track one, driving very slowly. I used my mouse to steer the vehicle and I had modified the tracking sensitivity to make it less prone to brutal steering. This avoided any brutal steering. I sometimes let the car dift sideways to inlude recoveries. I initially used center lane images only. Here is an example image of center lane driving:

![alt text][image1]

To augment the data set, I also flipped images and angles thinking that this would avoid any bias toward egative steering angles. For example, here is an image that has then been flipped:

![alt text][image2]
![alt text][image3]

To augment it further, and prevent the vehicle to slide to much sideways, I used the left and right cameras. The steering angle corresponding to the left camera was corrected by adding 0.2rad; while the steering angle corresponding to the left camera was corrected by removing 0.15rad. 

Output at this stage :

Train on 12638 samples, validate on 3160 samples
```sh
Epoch 1/4 : loss: 0.0363 - val_loss: 0.0370
Epoch 2/4 : loss: 0.0090 - val_loss: 0.0273
Epoch 3/4 : loss: 0.0069 - val_loss: 0.0268
Epoch 4/4 : loss: 0.0060 - val_loss: 0.0276
```
We can see that the model is greatly overfitting. At that point, I started to add dropout and to complexify the network as explained previously.

Finally, I added the example data provided in the project resources and added it to my own. With the three cameras, and the flipped images, the training set contains 51210 images. Pre-processing consist in normalizing images and cropping 50 pixels at the top and 20 at the bottom. At the beginning of each epoch, the training set is shuffled.

The output of the code at this point is the following 
```sh
Epoch 1/5 : loss: 0.0150 - val_loss: 0.0136
Epoch 2/5 : loss: 0.0122 - val_loss: 0.0120
Epoch 3/5 : loss: 0.0113 - val_loss: 0.0103
Epoch 4/5 : loss: 0.0108 - val_loss: 0.0098
Epoch 5/5 : loss: 0.0103 - val_loss: 0.0094
```
The evolution of the training and validation losses during the training is shown on the figure below.
![alt text][image4]