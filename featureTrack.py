
import cv2
import numpy as np


# create sift
sift = cv2.xfeatures2d.SIFT_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# get images
start = 90
img1 = cv2.imread('..\\images\\D3\\img' + str(start) + '.jpg',0)
img2 = cv2.imread('..\\images\\D3\\img' + str(start + 1) + '.jpg',0)
start = start + 1

# find and sort matches
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2, None)

kp1 = sift.detect(img1)
kp2 = sift.detect(img2)
kp1, des1 = brief.compute(img1,kp1)
kp2, des2 = brief.compute(img2,kp2)

bf = cv2.BFMatcher(crossCheck = True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)

# only match with one feature
kp = kp1[matches[63].queryIdx]
kp, des = brief.compute(img1, [kp])

m = bf.match(cv2.UMat(des),des2)
img3 = cv2.drawMatches(img1,kp,img2,kp2,m,outImg=None ,flags=2)

cv2.imshow('matches',img3)
cv2.waitKey(0)

# loop throught the rest of the pictures
while(True):
	#get same feature in next image
	kp = kp2[m[0].trainIdx]
	kp, des = brief.compute(img2, [kp])
	img1 = img2
	start = start + 1
	print(start)

	# get next image
	img2 = cv2.imread('..\\images\\D3\\img' + str(start) + '.jpg',0)
	kp2 = sift.detect(img2)
	kp2, des2 = brief.compute(img2,kp2)

	# plot match
	m = bf.match(cv2.UMat(des),des2)
	img3 = cv2.drawMatches(img1,kp,img2,kp2,m,outImg=None ,flags=2)

	cv2.imshow('matches',img3)
	k = cv2.waitKey(0)
	# esc key
	if k == 27:
		break

cv2.destroyAllWindows()