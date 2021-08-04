# Deep learning

Before we move on to our models, I would like to discuss why we chose deep
learning, instead of other traditional neural network models.

We can divided this topic into

- what image classification is
- what is deep learning, and
- why use deep learning for image classification

## image classification

When we use a computer to analyze and classify images into classes, it is know
as image classification.

## deep learning

Deep learning is a subset of machine learning in artificial intelligence that
has networks capable of learning unsupervised from data that is unstructured or
unlabeled. Also known as deep neural learning or deep neural network

## using deep learning for image classification

Since deep learning allows machines identify and extract features from images,
the algorithms and models can learning what features to look for while analyzing
lot of images.

# Models

## CNN

- cnn is a type of neural network model
- mostly used for image classification and recommendation systems.
- architecture of a cnn includes:
	- convolutional layer (parameters include the learnable filters, inputs are
	usually tensors)
	- pooling layer (a from of non-linear down sampling, reduces the dimension
	of data)
	- ReLU layer (rectilinear learning unit, which applies non-saturating
	activation function $f(x) = max(0, x)$), thus removing negative values from
	activation map by turning them to zero.
	- fully connected layer (neurons of this layer have full connection to all
	activations in the previous layer)
	- loss layer (how training penalizes the deactivation between the predicted
	output and the true data label)
	- Dropout layer (is a technique where randomly selected neurons are ignored during
	training. They are "dropped-out" randomly.)

image of cnn's architecture

## VGG 16

- is cnn model
- is a pre trained model (imagenet dataset)
- winner of the 2014 imagenet competition
- achieved upto $92.7$% accuracy in the imagenet dataset
- uses 3x3 kernel sized filters
- was trained on multiple nvidia titan black gpu's

image of vgg 16's architecture

## About the vgg's architecture

In this picture the layers in

- black are convolutional + ReLU layer
- red are the max pooling layer
- cyan are the full connected + ReLU layer
- yellow is the softmax layer.

## ResNet 50

- another type of cnn model
- also a pre trained model (imagenet dataset)
- winner of the 2015 imagenet competition
- achieved upto 84.4%
- mainly consists of residual learning blocks
- resnet 50 is nothing but a 50 layer residual learning network.

image of  resnet 50's architecture

## About the resnet's architecture

5th column is for the resnet layer

The ResNet-50 model consists of 5 stages each with a convolution and Identity block
It has 48 convolutional layers, 1 MaxPool layer and 1 average pool layer

Next slide is implementation by rajarshi
