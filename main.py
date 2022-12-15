#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Train the model
# After training we will have a dataset
# dataset is sent to CNN classifier we have used (VGG16) here.
# Examples of other CNN classifier: Resnet, InceptionNet, MObile Net V2
# Mobile Net V2 is very light, It uses Rpi, 25 Mb, accuray is 70% wrt 72% of VGG16 whih uses 500 Mb


# In[2]:


# Mask problem is easy as compared to emotion problem.
# Here, Model is exported. 
# We are using transfer learning, where last layer is removed and our layer is added.
# Input is taken using webcam that is applied to the model.
# Screen detection is done first.
# We are cropping the face only with the rectangle for improved accuracy.
# Haar cascade is used to get frontal face detection.


# In[3]:


import os
# to get all files


# In[4]:


from keras.preprocessing import image
import cv2


# In[5]:


categories = ['without_mask','with_mask']


# In[6]:


data = []
for category in categories:
    path = os.path.join('train',category)
    
    label = categories.index(category)
    # without_mask index is 0
    # with_mask index is 1
    
    for file in os.listdir(path):
        # Now we will read image using opencv
        
        img_path = os.path.join(path,file)
        img = cv2.imread(img_path)
        img = cv2.resize(img,(224,224))
        
        data.append([img,label]) # for adding both the with and without mask images
        # (image,0) for image in without_mask folder
                                 #(image,1) for image in with_mask folder
        
# we will create a dataset 2 columns, 1st one is image and 2d is the label   


# In[7]:


len(data)


# In[8]:


import random


# In[9]:


random.shuffle(data) # for mixing the with and without mask images


# In[10]:


# to separate X and y and then convert it into numpy array
X = []
y = []

for features,label in data:
    X.append(features)
    y.append(label)


# In[11]:


len(X)


# In[12]:


len(y)


# In[13]:


import numpy as np


# In[14]:


X = np.array(X)
# X should be 2d array
y = np.array(y)
# y should be 1d array since it is just a label


# In[15]:


X.shape
# 1508 is no of images with its shape (224,224,3) 224x224 and 3 is no of channels


# In[16]:


y.shape
# y are the labels 


# In[17]:


y


# In[18]:


X


# In[19]:


X = X/255# in order to standize the X


# In[20]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[21]:


X_test.shape


# In[22]:


# 302 is no of images in test


# In[23]:


# we will create CNN architecture and do the transfer learning


# In[24]:


from keras.applications.vgg16 import VGG16


# In[25]:


vgg = VGG16()


# In[26]:


vgg.summary()
# This is our vgg model


# In[27]:


# predictions (Dense)         (None, 1000)              4097000
#This last layer will be removed and we will add our layer with 1 node


# In[28]:


from keras import Sequential


# In[29]:


model = Sequential()


# In[30]:


for layer in vgg.layers[:-1]:  # to exclude last layer
    model.add(layer)


# In[31]:


model.summary()
# now last layer is removed and new model is created
# vgg is a functional model, but we needed sequential thats why we used sequential workflow


# In[32]:


#we will freeze all the layers so that their weights wont be updated during training 


# In[33]:


for layer in model.layers:
    layer.trainable = False


# In[34]:


model.summary()


# In[35]:


# Now, all trainable parameters are set to 0 i.e. parameters are freezed


# In[36]:


# we are adding our last layer
from keras.layers import Dense


# In[37]:


model.add(Dense(1,activation='sigmoid'))


# In[38]:


model.summary()


# In[39]:


# this last layer has 4097 parameters, its 1 and 4096 coming from previous layer


# In[40]:


# 4097 are trainable parameters


# In[41]:


model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
# to increase the accuracy, we can use learning rate of 0.0001


# In[ ]:


model.fit(X_train,y_train,epochs=5,validation_data=(X_test,y_test))


# In[ ]:


cap = cv2.VideoCapture(0) # to detect the face, 0 for webcam


# In[ ]:


while True:
    
    ret,frame = cap.read()
    #reading the image
    #ret gives us true or flase value
    #frame gives us the image
    
    # call for detection method
    img = cv2.resize(frame,(224,224))
    
    y_pred = detect_face_mask(img) # image is sent to the function, the ouput stored in y_pred
    # frame to get a smaller video, img for the normal window size
    
    coods = detect_face(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
    
    for x,y,w,h in coods:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) # 1is thickness
    
    if y_pred == 0: 
        draw_label(frame,"Mask",(30,30),(0,255,0))
    else:
        draw_label(frame,"No Mask",(30,30),(0,0,255))
        
    
    draw_label(frame,"Face Mask Detection", (10,10),(255,0,0))
    #(255,0,0) is blue color
    
    print(y_pred)
    
    cv2.imshow('windows', frame) # we are displaying the frame
    
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
        
cv2.destroyAllWindows()


# In[ ]:


def detect_face_mask(img):
    
    #y_pred = model.predict_classes(img.reshape(1,224,224,3))
    y_pred = model.predict(img.reshape(1,224,224,3))
    y_pred = np.round(y_pred).astype(int)
    # our trained model will predict whether it is with or without mask as 1 or 0 as output
    
    return y_pred


# In[ ]:


sample1 = cv2.imread('sample1.jpg')
sample1 = cv2.resize(sample1,(224,224))


# In[ ]:


detect_face_mask(sample1)


# In[ ]:


# 1 index means with mask, which is right.


# In[ ]:


# now implementing a text in the video telling us whether mask is on or not

def draw_label(img,text,pos,bg_color): #pos is position
    
    
    text_size = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,cv2.FILLED) # 1 is font scale
    # cv2.filled is thickness
    
    end_x = pos[0]+text_size[0][0] + 2 # 2 is the margin
    end_y = pos[1]+text_size[0][1] - 2
    
    cv2.rectangle(img,pos,(end_x,end_y),bg_color,cv2.FILLED)
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1,cv2.LINE_AA)
    # (0,0,0) is the color, LINE_AA is type of line used
    
    


# In[ ]:


#face detection square

#import haar cascading file

haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(img):
    
    coods = haar.detectMultiScale(img)
    return coods




