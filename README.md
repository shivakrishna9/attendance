# attendance
Automatic attendance monitoring system - cs231n Project


Now I know this is not upto the mark, but what I need to implement is a faster R-CNN for face detection and on top of it a fully connected layer for classification of faces into people. But first, I need a dataset of images and a pre-trained model of face detection as well as face recognition. Maybe the former is easy to implement, but the latter is just maybe an FC6,7,8 so must not take a lot of time to get done. However, I need to get somethings first, which are more important with respect to what I am gonna use. 73 classes of students who are going to be classified, as the number of classes increases, fine tuning might take a plow, but I guess we're gonna do just fine.

As a fun fact, I have no idea where to start, though I do know that I need to figure it out as quickly as possible. I have a pre-trained model, but I have doubts regarding if I'd be able to use it in my own architecture. Obviously not! But then I do need to implement some architecture fast enough to be able to train it with both detection as well as recognition.


Error right now:
Exception: TypeError while preparing batch. If using HDF5 input data, pass shuffle="batch".