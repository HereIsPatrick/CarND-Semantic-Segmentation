# Semantic Segmentation

[image1]: images/1.png
[image2]: images/2.png 
[image3]: images/3.png 
[image4]: images/4.png 
[image5]: images/5.png
[image6]: images/6.png 
[image7]: images/7.png 
[image8]: images/8.png
 
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN). To construct Semantic Segmentation(Fully convolutional networks base on VGG16, use transfer learning that reduce training time.

### Demo Video
[![Path Planning](http://img.youtube.com/vi/eDj3TE-x6-w/0.jpg)](https://www.youtube.com/watch?v=eDj3TE-x6-w
 "Semantic Segmentation")

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Convolutional network architecture and code
The code(main.py) download pre trained VGG16 model, get the input layer, keep parameter of layer 3, 4 and 7. You can see convolution network layer define as below. 

	# Step. 1x1 convolution of vgg layer 7
    layer7a_out = tf.layers.conv2d(vgg_layer7_out,
                                   num_classes,
                                   1, # kernel size
                                   padding= 'same',
                                   kernel_initializer= tf.random_normal_initializer(stddev=weight_stddev),
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(weight_regularized))
    # Step. upsample
    layer4a_in1 = tf.layers.conv2d_transpose(layer7a_out,
                                             num_classes,
                                             4, # kernel size
                                             strides= (2, 2),
                                             padding= 'same',
                                             kernel_initializer= tf.random_normal_initializer(stddev=weight_stddev),
                                             kernel_regularizer= tf.contrib.layers.l2_regularizer(weight_regularized))
    # Step. 1x1 convolution of vgg layer 4
    layer4a_in2 = tf.layers.conv2d(vgg_layer4_out,
                                   num_classes,
                                   1, # kernel size
                                   padding= 'same',
                                   kernel_initializer= tf.random_normal_initializer(stddev=weight_stddev),
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(weight_regularized))
    # Step. Skip layer
    layer4a_out = tf.add(layer4a_in1, layer4a_in2)
    # upsample
    layer3a_in1 = tf.layers.conv2d_transpose(layer4a_out, num_classes,
                                             4,
                                             strides= (2, 2),
                                             padding= 'same',
                                             kernel_initializer= tf.random_normal_initializer(stddev=weight_stddev),
                                             kernel_regularizer= tf.contrib.layers.l2_regularizer(weight_regularized))
    # Step. 1x1 convolution of vgg layer 3
    layer3a_in2 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1,
                                   padding= 'same',
                                   kernel_initializer= tf.random_normal_initializer(stddev=weight_stddev),
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(weight_regularized))
    # Step. Skip connection
    layer3a_out = tf.add(layer3a_in1, layer3a_in2)

    # Step. Deconvolution
    last_layer = tf.layers.conv2d_transpose(layer3a_out, num_classes, 16,
                                               strides= (8, 8),
                                               padding= 'same',
                                               kernel_initializer= tf.random_normal_initializer(stddev=weight_stddev),
                                               kernel_regularizer= tf.contrib.layers.l2_regularizer(weight_regularized))

Optimize use Adam optimizer.

### Traing Parameter
- keep_prob:0.5
-  learning_rate: 0.0001
-  epochs: 80
-  batch_size: 5

### Training Result
The model decrease loss over time

- 10 epochs: 0.087
- 20 epochs: 0.063
- 40 epochs: 0.026
- 80 epochs: 0.007

correct label the road that label at least 80% of the road, and no more than 20% of non-road pixels as road.

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [post](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
