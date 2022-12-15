# Mask-Detection-
Mask Detection using CNN

# Steps
Train the model.
After training we will have a dataset.
Dataset is sent to CNN classifier, we have used (VGG16) here.
Examples of other CNN classifier: Resnet, InceptionNet, MObile Net V2.
Mobile Net V2 is very light, It uses Rpi, 25 Mb, accuray is 70% wrt 72% of VGG16 whih uses 500 Mb.

Here, Model is exported. 
We are using transfer learning, where last layer is removed and our layer is added.
Input is taken using webcam that is applied to the model.
Screen detection is done first.
We are cropping the face only with the rectangle for improved accuracy.
Haar cascade is used to get frontal face detection.
