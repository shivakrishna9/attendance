# attendance
Automatic attendance monitoring system - CS Lab Jamia MIllia Islamia

# What You See:

So, this was the final year project for the completion of my Bachelor's Degree in Jamia Millia Islamia, Delhi, India.

This project has a few parts. The *recognise* folder contains the files which run the VGG, ResNet~35 and Resnet~50 Networks. These networks were implemented because of the fact that they have proven to be the best in image recognition competitions and have minimum top-five errors.

The *website* folder contains the website that was developed to showcase this. The website has been developed using Django and uses an html template from some website online.

# How it works:

There are a few things that do work, some do not. In any case, if someone does try to improve over this, you would require, some specific libraries like Keras, Numpy, Pandas, Django. These have been used extensively. Keras was used with Tensorflow as well as Theano which again will be required.

Things that do not work include the ResNet~50. ResNet-50 should only be run if and only if you have a system with GPU memory >= 4GB, otherwise with Caffe it will definitely create problems.
With Theano, with GPU memory == 4GB it is easily loaded on the system, but you will not be able to do anything else while it is running. 
With TensorFlow, with GPU memory == 4GB it runs, but unstably, i.e. it is going to run, but after some epochs it fails. This may be due to a bug in TensorFlow or a bug in my implementation of something.

Usually everything starts running by running the **demo.py** file which runs on any image in the *website/demo/* folder (you might have to look into the paths once).

The whole project now seems a waste of time since I spent a whole Spring (2016) learning and implementing everything possible by an undergrad only to find that I did not have ample resources to complete it and the project did not run real-time. To make it real-time, other methods, detectors and recognisers, which take much less time need to be implemented.

# DONE

Train VGGNet on a very low learning rate ! --running

# TODO

Unfortunately, I do not have enough know-how to further improve the project and hence I have taken a step forward to complete Masters Degree in Computer Science and then (probably and possibly) Ph.D. from a renowned university in The United States of America. Till then, I would leave this as it is.

1. Apply faster-RCNN to the model in maybe two ways :
    
    *   run haar cascade after the conv network

    *   implement region proposal networks

2. Take someone with you now ?

3. Implement it in your house ??


# Everyone is encouraged to contribute.
