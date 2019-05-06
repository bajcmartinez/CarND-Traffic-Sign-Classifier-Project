## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
Build a system to classify german traffic signs 

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[ev_distribution]: ./assets/distribution.png "Distribution"
[p_sample]: ./assets/sample.png "Prediction Sample"
[o_layer1]: ./assets/layer_1.png "Layer 1"
[o_layer2]: ./assets/layer_2.png "Layer 2"

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

```
Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
```

#### 2. Exploratory visualization of the dataset.

First to understand the dataset I've plotted on a chart the distribution of samples per class (traffic sign type)

![alt text][ev_distribution]

The dataset is distribution is not very even, so we will have to work on the dataset to even the distribution, that will help our network train better.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

This process was the result of a lot of experimentation.

1. I had to decide whether to work with color or grayscale images. Since I couldn't perceive major differences on the performance of the network from one to another I decided to work with grayscale, as it reduces the number of input nodes significally and thus translating in much faster processing of the network   

2. Image augmentation, as we saw in the exploratory visualization we needed to improve the samples of our dataset, so we did it by generating images from sample images and applying random transformations. The supported transformations are:
    2.1 Image lighting using LAB space and CLAHE over L channel
    2.2 Vertical flip
    2.3 Image rotation
    2.4 Image translation
    
3. Image normalization, convert the values of the matrix into values closer to -1 and 1 instead of 255 scale

After augmenting our sample set contained 167907. With even higher samples performance was inclining towards better results, however training time was growing and had to cut off for practicality. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

```
TrafficSignNet Model Summary
-------------------------------------------------------------
Layer      type       in_depth   out_depth  filter_size
1          conv       1          32         5         
2          conv       32         64         5         
3          conv       64         128        5         
4          dense      20480      240        None      
5          dense      240        43         None      
-------------------------------------------------------------
```


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Finding the right parameters for the model was the result of experimenting with the dataset, and different type of models.

In essence the final result is based out of convolution layers that are built like the following:

```python
def _conv(self, x, out_depth, filter_size=5, max_pooling=True, dropout=True):
    """
    Creates a convolution layer
    
    :param x: 
    :param out_depth: 
    :param filter_size: 
    :return: convolution layer
    """
    self.layer_count += 1
    self.model_stack.append({
        'layer': self.layer_count,
        'type': 'conv',
        'in_depth': x.get_shape().as_list()[-1],
        'out_depth': out_depth,
        'filter_size': 5
    })
    
    conv_w = tf.Variable(tf.truncated_normal(shape=(filter_size, filter_size, x.get_shape().as_list()[-1], out_depth), mean=self.mu, stddev=self.sigma))
    conv_b = tf.Variable(tf.zeros(out_depth))
    conv = tf.nn.conv2d(x, conv_w, strides=[1, 1, 1, 1], padding='SAME') + conv_b
    conv = tf.nn.relu(conv, name="layer_{0}".format(self.layer_count))
    conv = tf.layers.batch_normalization(conv)
    
    if max_pooling:
        # Max pooling 
        conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    if dropout:
        # Dropout
        conv = tf.nn.dropout(conv, keep_prob=self.conv_keep_prob) 
    
    return conv
```

Each convolution layer applies `relu` activation function plus batch normaliation as well as the option for max pooling and dropout.

For dense layers:

```python
def _dense(self, x, units, last=False):
    """
    Creates a dense layer
    
    :param x: 
    :param units: 
    :return: dense layer
    """
    self.layer_count += 1
    self.model_stack.append({
        'layer': self.layer_count,
        'type': 'dense',
        'in_depth': x.get_shape().as_list()[-1],
        'out_depth': units,
        'filter_size': None
    })
    
    dense_w = tf.Variable(tf.truncated_normal(shape=(x.get_shape().as_list()[-1], units), mean=self.mu, stddev=self.sigma)) 
    dense_b = tf.Variable(tf.zeros(units))
    dense = tf.matmul(x, dense_w) + dense_b
    if last:
        return dense
    
    dense = tf.nn.relu(dense, name="layer_{0}".format(self.layer_count))
    dense = tf.layers.batch_normalization(dense)
    
    return tf.nn.dropout(dense, keep_prob=self.dense_keep_prob)
```

Each dense layers is built on top of the matmul function, with `relu` activation function, batch normalization and optional dropout.

As per the model training, I decided to go with `Adam` optimizer, which allows me to easily update the learning rate according to the decay parameters.

For the network parameters:

```python
model = TrafficSignNet(learning_rate=0.0001, conv_keep_prob=0.9, dense_keep_prob=0.5, n_classes=n_classes)
model.train(x_train_augmented, y_train_augmented, X_valid_sample, y_valid, epochs=200, batch_size=512)
```

Those worked the best, using drop outs with low rate for convolutions and 50% on dense.
The learning rate wordked best with smaller numbers, but by reducing the learning rate I had to increase the number of epochs to make the network train correctly. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.9995
* validation set accuracy of 0.9358
* test set accuracy of 0.9345

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The first architecture was more simple, having no dropouts, no batch normalization, fewer layers, and it was evolving as I was researching and experimenting.

* What were some problems with the initial architecture?
Actually not much was wrong, the results were not very bad, 91% accuracy on test dataset, but to push a 1% increment required a lot of effort and time experimenting. 
 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I've downloaded 5 images from the web and run the prediction model, here are the results

![alt text][p_sample]


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

This has been very fun, seeing how the model actually works on each convolution layer, and understanding how the computer "sees" the information.

In the first layer we can still see something that resembles the picture  
![alt text][o_layer1]

But when we get to the second convolution network, it's not clear anymore for us.
![alt text][o_layer2]