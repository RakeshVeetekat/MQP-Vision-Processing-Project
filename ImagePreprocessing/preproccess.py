import numpy as np
import cv2
from matplotlib import pyplot as plt


img1 = cv2.imread('img1652.jpg')
print('Original size',img1.shape)
cv2.imshow('original', img1)


# setting dim of the resize
height = 800
width = 800
dim = (width, height)
res = cv2.resize(img1, dim, interpolation=cv2.INTER_LINEAR)

cv2.imwrite('img1652.png',res)

# Checking the size
print("RESIZED", res.shape)
    
res = img1

# Remove noise
# Gaussian
blur = cv2.GaussianBlur(res, (5, 5), 0)

image = blur
cv2.imshow('Gaussian Blur', image)


# Gray Scale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

cv2.imshow('gray scale', gray)

#Histogram Normalization
dst = cv2.equalizeHist(gray)
cv2.imshow('histogram normailization', dst)
cv2.imwrite('img1652-preprocessed.png',dst)

cv2.waitKey(0)
