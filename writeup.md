# **Traffic Sign Recognition** 

## Writeup

### Chuan Yu's writeup file for Udacity Self-Driving Car Nanodegree Program Project 2 Traffic Sign Classifier

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

[image1]: ./examples/AllSignImages.png "AllSignImages"
[image2]: ./examples/TrainningDataSignCounts.png "TrainningDataSignCounts"
[image3]: ./examples/Normalize.png "Normalize"
[image4]: ./new_images_1/13_Yield.png "Traffic Sign 1"
[image5]: ./new_images_1/31_Wild_animals_crossing.png "Traffic Sign 2"
[image6]: ./new_images_1/37_Go_straight_or_left.png "Traffic Sign 3"
[image7]: ./new_images_1/3_Speed_limit_60kph.png "Traffic Sign 4"
[image8]: ./new_images_1/9_No_passing.png "Traffic Sign 5"
[image9]: ./new_images/13_Yield.png "Traffic Sign 6"
[image10]: ./new_images/27_Pedestrians.png "Traffic Sign 7"
[image11]: ./new_images/31_Wild_animals_crossing.png "Traffic Sign 8"
[image12]: ./new_images/37_Go_straight_or_left.png "Traffic Sign 9"
[image13]: ./new_images/8_Speed_limit_120kph.png "Traffic Sign 10"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/GitHubChuanYu/Project2_TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration

#### 1. A basic summary of the data set is provided using basic python code:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

I use two ways to visualize the dataset. One is plotting all the 43 types of sign images with class and name. Another is a vertical bar chart showing how many images we have for each kind of sign in training data set. The visaulized images are shown below:

![alt text][image1]
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decide not to convert the original color image to grayscale image for preprocessing of the dataset because CNN model can take care of several layers of color images with a flatten layer.

Instead I decide to normalize the data because as mentioned in the class, a normalized input data is good for training neural network. The detailed reason is neural network training is a gradient descent kind of optimization process, in order to make sure the optimization coverges to the minium value of error function, a normalized input with zero mean and equal variance will really be helpful.

Here is an example of a traffic sign image before and after normalization.

![alt text][image3]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Layer1: Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|	outputs 28x28x6											|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6 				|
| Layer2: Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16  |
| RELU					|	outputs 10x10x6											|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16 				|
| Flatten		| input 5x5x6, output 5x5x16=400       									|
| Layer3: Fully connected		| outputs 300        									|
| RELU					|	outputs 300											|
| Dropout				| Keep_prob = 0.5 for training and 1 for validation, outputs 300        									|
|	Layer4: Fully Connected		|				input 300, output 172								|
|	RELU					|				outputs 172								|
| Dropout				| Keep_prob = 0.5 for training and 1 for validation, outputs 172        									|
|	Layer5: Fully Connected		|				input 172, output 43								|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the default training code from LeNet-5 implementation shown in class. I tried different parameters, finally found out these parameters can give me a good result:
* Epochs = 20
* Batch szie = 128
* Learning rate = 0.001
* Keep_prob for dropout = 0.5 for training, and 1 for validation and test

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.955 
* test set accuracy of 0.946

I did improve the training model with several interations.

Firstly, I choose the LeNet-5 implementation as suggested. The reasons why I choose it are:
* It is good for classifying images as mentioned in class.
* It is also tested and verified to have a good start accuracy in class.
* Its CNN has unique feature which can detect unique patterns in images regardless of location.

However, I found a main problem for the initial LeNet-5 implementation that it has a very high training accuracy, and also very low validation accuracy. This is a typical overfitting as mentioned in notes. So I tried two ways to improve it:
* Using dropout
* Increase the output size of last fully connected layer so that it contains more information for final classifier with 43 classes compared with previous 10 classes in original LeNet-5 implementation.

I tried dropout on different layers, it seems like it has different effects when applied on different layers. I have tried dropouts on first two convolutional layers, it seem not very effective with still high accuracy on training and low accuracy on validation. And then I tried the dropouts on last to fully connected layers, it seems to help a lot. So finally I decide to apply two dropouts on last two fully connected layers. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I have downed two sets of five German traffic sign images from web:

* First set of five images are:

![alt text][image9] ![alt text][image10] ![alt text][image11] 
![alt text][image12] ![alt text][image13]

* Second set of five images are:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

**Note: For curiosity, I have downloaded the "Pedestrians" and "120km/h speed limit" signs with some rotation (the sign inside the image is not parallel with the image surface) in first set. My guess is that this unique feauture of these two sign images will make them a little little bit difficult for my trained CNN model to classify if all the images inside the training dataset do not have this sign angle feature to the image surface.**

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

* I have tried two sets of five new traffic sign images I download from web. 

* The first set is in folder [new_images](https://github.com/GitHubChuanYu/Project2_TrafficSignClassifier/tree/master/new_images). Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Wild animals crossing      		| Wild animals crossing   									| 
| Pedestrians     			| Go straight or left 										|
| Go straight or left 					| Go straight or left 											|
| 120 km/h	      		| 20km/h					 				|
| Yield      | Yield     							|

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This seems not match the high accuracy on the test set of 0.946. The result is shown in [Traffic_Sign_Classifier_1.html](https://github.com/GitHubChuanYu/Project2_TrafficSignClassifier/blob/master/Traffic_Sign_Classifier_1.html)

* The second set is in folder [new_images_1](https://github.com/GitHubChuanYu/Project2_TrafficSignClassifier/tree/master/new_images_1). Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Wild animals crossing      		| Wild animals crossing   									| 
| No passing     			| No passing 										|
| Go straight or left 					| Go straight or left 											|
| 60 km/h	      		| 60km/h					 				|
| Yield      | Yield     							|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This seems match the high accuracy on the test set of 0.946. The result is shown in [Traffic_Sign_Classifier_Final.html](https://github.com/GitHubChuanYu/Project2_TrafficSignClassifier/blob/master/Traffic_Sign_Classifier_Final.html)

* Based on the test results of two sets of new sign images, it seems like some specific sign images like '120km/h' and 'pedestrians' are hard to be predicted correctly using current model. This matches my initial guess when I purposely chose two images with the sign having an angle with the image surface instead of parallel to the image surfaces like the rest of chosen images. The unique two signs are not classified, whereas the rest normal signs are classified. This makes me think that maybe the original training dataset should be augmented with some features like rotating the sign to make it have angle to the image surface. Then our CNN training model are trained with this augmented dataset, then maybe the unique "120km/h" and "pedestrians" will be classified correctly with the new model trained by augmented dataset.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

* The softmax probabilities for the predictions of two sets of five new sign images are also documented in two html files shown above.

* For the first set in in folder [new_images](https://github.com/GitHubChuanYu/Project2_TrafficSignClassifier/tree/master/new_images). The softmax probability results are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Wild animals crossing   									| 
| 0     				| Pedestrians 										|
| 1					| Go straight or left											|
| 0	      			| 120km/h					 				|
| 1				    | Yield      							|

* For the second set in in folder [new_images_1](https://github.com/GitHubChuanYu/Project2_TrafficSignClassifier/tree/master/new_images_1). The softmax probability results are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Wild animals crossing   									| 
| 1     				| No passing 										|
| 1					| Go straight or left											|
| 1	      			| 60km/h					 				|
| 1				    | Yield      							|

It seems like for 'Pedestirans' and '120km/h' sign images, the model cannot predict it.

### Possible improvements
One possible improvement would be to augamment the training dataset with more features of the sign inside the image like rotation of the sign to make it have an angle to the image surface so that the CNN model trained with augamented dataset can identify more unique and realistic sign images like what I picked here for "120km/h speed limit" and "pedestrians" successfully.
