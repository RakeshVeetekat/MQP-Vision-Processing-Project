
import cv2
import numpy as np


# create sift
sift = cv2.xfeatures2d.SIFT_create()

#stores original point to compare to
orgpt = 0

# get images
start = 90
img1 = cv2.imread('..\\images\\D3\\img' + str(start) + '.jpg',0)
img2 = cv2.imread('..\\images\\D3\\img' + str(start + 1) + '.jpg',0)
start = start + 1

# find and sort matches
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
	if m.distance < 0.75*n.distance:
	    good.append(m)

# only match with one feature
#115, 115
kp = kp1[good[115].queryIdx]
kp, des = sift.compute(img1, [kp])

m = bf.knnMatch(cv2.UMat(des),des2, k=4)


#sort matches in terms of distance from original

orgpt = kp[0].pt
minPoint = kp[0].pt
minDis = 10000
number = 0
for i in range(0,4):
	#get distance to orgpt
	pt = kp2[m[0][i].trainIdx]
	dis = (orgpt[0] - pt.pt[0])**2 + (orgpt[1] - pt.pt[1])**2

	#if closer that min point
	if(dis<minDis):
	#set equal to minpoint
		minDis = dis
		number = i
		minpoint = pt


img3 = cv2.drawMatches(img1,kp,img2,kp2,[m[0][number]],matchColor = (0,255,0), outImg=None ,flags=2)

cv2.imshow('matches',img3)
cv2.waitKey(0)

# loop throught the rest of the pictures
while(True):
	#get same feature in next image
	kp = kp2[m[0][number].trainIdx]
	kp, des = sift.compute(img2, [kp])
	img1 = img2
	start = start + 1
	print(start)

	# get next image
	img2 = cv2.imread('..\\images\\D3\\img' + str(start) + '.jpg',0)
	kp2, des2 = sift.detectAndCompute(img2, None)

	# plot match
	m = bf.knnMatch(cv2.UMat(des),des2, k=4)

	orgpt = kp[0].pt
	minPoint = kp[0].pt
	minDis = 10000
	number = 0
	for i in range(0,4):
		#get distance to orgpt
		pt = kp2[m[0][i].trainIdx]
		dis = (orgpt[0] - pt.pt[0])**2 + (orgpt[1] - pt.pt[1])**2

		#if closer that min point
		if(dis<minDis):
		#set equal to minpoint
			minDis = dis
			number = i
			minpoint = pt

	img3 = cv2.drawMatches(img1,kp,img2,kp2,[m[0][number]],matchColor = (0,255,0),outImg=None ,flags=2)

	cv2.imshow('matches',img3)
	k = cv2.waitKey(0)
	# esc key
	if k == 27:
		break

cv2.destroyAllWindows()


# def getDistance(m):
# 	point = m
