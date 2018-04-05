# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image4]: ./DataSets/traffic-signs-secondary/speed30_sign.jpg "Traffic Sign 1"
[image5]: ./DataSets/traffic-signs-secondary/no-passing-sing.jpg "Traffic Sign 2"
[image6]: ./DataSets/traffic-signs-secondary/stop-sign.jpg "Traffic Sign 3"
[image7]: ./DataSets/traffic-signs-secondary/right-of-way.jpg "Traffic Sign 4"
[image8]: ./DataSets/traffic-signs-secondary/do-not-enter.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/bflyth/Traffic-Classifier)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
* The size of the validation set is ?
* The size of test set is ?
* The shape of a traffic sign image is ?
* The number of unique classes/labels in the data set is ?

#### 2. Include an exploratory visualization of the dataset.

Within the jupyter notebook is a bar chart showing how many images exist per label. The next visualization is after a preprocess. 


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because this would effectively flatten the image to a 32 x 32 x 1 rather than 32 x 32 x 3.

As a last step, I normalized the image data using cv2.equalizeHist() because it normalized the bins to a range of 255 and then calculates an integral, then running a transform based on the integral.

I did not choose to run added random data.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x5 	|
| RELU					|												|
| DROPOUT	        	| .65 chance of dropout 				        |
| Max Pooling           | outputs 14x14x6                               |
| Convolution 5x5     	| 6x6 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| DROPOUT	        	| .65 chance of dropout 				        |
| Max Pooling           | outputs 5x5x16        						|
| Flatten				| outputs 400									|
| Fully Connected Layer | outputs 120									|
| RELU					|												|
| DROPOUT	        	| .65 chance of dropout 				        |
| Fully Connected Layer | outputs 84									|
| RELU					|												|
| DROPOUT	        	| .65 chance of dropout 				        |
| Fully Connected Layer | outputs number of classes (43)				|
| Regularization        | algorithm runs regularization with L1         |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I am using the adamoptimizer as it will converge faster than a gradient descent with less tuning (according to https://stats.stackexchange.com/questions/184448/difference-between-gradientdescentoptimizer-and-adamoptimizer-tensorflow)

My batch size is set to 128 images. I played with the learning rate using .0001, .0005, .0007 and finally .0008 seems to converge in a timely manner without losing accuracy. Dropout for all events is set to a standard 65% chance. This was run with 150 Epochs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98.9%
* validation set accuracy of 96%
* test set accuracy of 93.2%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
    At first I attempted to run 3 convolutions followed by a flatten layer and then two FC layers. I thought this would result in a small array with a high depth (third dimension), but instead the model took an absurd amount of time to run, and did not quite give results. That model did not break 85%

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    I removed one of the three convolutions after reading various models online from published reports following the LeNet. This definitely allowed my network to run in a more timely manner, and also resulted in greater accuracy. I still see that my model is overfitting, with a 2.9% drop in accuracy between training and validation, but 96% is still over the .93 threshold.
    
* Which parameters were tuned? How were they adjusted and why?
    My main focus was to adjust the learning rate and Epoch count. I figure these would have the highest impact on model efficiency by either allowing from broader steps and less epochs, or finer steps and longer epochs. The result was to choose a slightly broader step (from .0001 to .0008) and a moderate length of Epochs (from starting with 250 and dropping to 50, I chose 150)
    
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    Convolutional layers work well to start this model off by extracting features from the image. The goal of this project is to extract features from street signs to determine what the sign means, and so by convolving the images, the model can get an idea of important features in the image as the filter scans through. Dropouts help to scramble the feauture map and avoid the model from overfitting to the given data. By removing random channels, the filters lose bits of information allowing the network to learn better with slightly less data.

If a well known architecture was chosen:
* What architecture was chosen?
    LeNet
* Why did you believe it would be relevant to the traffic sign application?
    LeNet has been known since the 90's to extract feautures efficently.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    The final test accuracy was just over .93, which was this projects required accuracy for validation. Validation scores reached upwards of 98% but this final training session with the chosen parameters resulted in 96% validation accuracy.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The images with a busy background such as trees or cars would possibly pose a threat to the models ability to classify them. Even after normalization these features could interfere with more important data to be extracted.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| General Caution								| 
| No Passing     		| No Passing 									|
| 30 km/h				| Vehicles over 3.5 metric tons prohibited		|
| Right of Way     		| Right of Way					 				|
| Do Not Enter			| Do Not Enter      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is disappointing and is not similar to the accuracy on the test set of 93%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.

For the first image, the model is absolutely sure that this is a General Caution sign (probability of 1.0), and the image actually contains a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00        			| General Caution								| 
| 0.0     				| 20km/h 										|
| 0.0					| 30km/h					          			|
| 0.0       			| 50km/h			        	 				|
| 0.0				    | 70km/h             							|

For the second image, the model is absolutely sure this is a No Passing sign (probability of 1.0), and the image is indeed a No Passing sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00        			| No Passing   									| 
| 0.0     				| 20km/h 										|
| 0.0					| 30km/h					          			|
| 0.0       			| 50km/h			        	 				|
| 0.0				    | 70km/h            							|

For the third image, the model is almost absolutely sure this is a Vehicles over 3.5 metric tons prohibited (probability of ~1.0), and the image is actually a speed sign for 30km/h.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ~1.00        			| Vehicles over 3.5 metric tons prohibited		| 
| 7.07624848e-10     	| 20km/h 										|
| 1.88525085e-13     	| No passing for vehicles over 3.5 metric tons  |
| 1.97141558e-17		| Go straight or left                           |
| 1.51753620e-29	    | Roundabout Mandatory 							|


For the fourth image, the model is absolutely sure this is a Right of Way sign (probability of 1.0), and the image is indeed a Right of Way sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00        			| Right of Way 									| 
| 0.0     				| 20km/h 										|
| 0.0					| 30km/h					          			|
| 0.0       			| 50km/h			        	 				|
| 0.0				    | 70km/h            							|


For the fifth image, the model is absolutely sure this is a Do Not Enter sign (probability of 1.0), and the image is indeed a Do Not Enter sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00        			| Do Not Enter 									| 
| 0.0     				| 20km/h 										|
| 0.0					| 30km/h					          			|
| 0.0       			| 50km/h			        	 				|
| 0.0				    | 70km/h            							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
