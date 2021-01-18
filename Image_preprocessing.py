#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('img1652GE.PNG')
print('Original size',img1.shape)
plt.imshow(img1)


# In[2]:


# setting dim of the resize
height = 220
width = 220
dim = (width, height)
res = cv2.resize(img1, dim, interpolation=cv2.INTER_LINEAR)

# Checking the size
print("RESIZED", res.shape)
    
# Visualizing one of the images in the array
original = res


# In[3]:


# ----------------------------------
# Remove noise
# Gaussian

blur = cv2.GaussianBlur(res, (5, 5), 0)

image = blur
# display(img1, image, 'Original', 'Blured')
plt.imshow(image)
#---------------------------------


# In[20]:


# Gray Scale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.imshow(gray)


# In[21]:


#Histogram Normailization

dst = cv2.equalizeHist(gray)
# cv2.imshow('Source image', src)
cv2.imshow('Equalized Image', dst)

plt.imshow(gray)


# In[22]:


plt.imshow(dst)
cv2.waitKey(0)

# In[ ]:




