import cv2
import numpy as np


def runShitomasi(img1,img2, description):

	#create breif descriptor
	brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

	# img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
	# img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

	# find corners
	corners1 = cv2.goodFeaturesToTrack(img1,1000,0.01,10)
	corners1 = np.int0(corners1)

	corners2 = cv2.goodFeaturesToTrack(img2,1000,0.01,10)
	corners2 = np.int0(corners2)

	# create keypoints
	kp1 = []
	for i in corners1:
		x,y = i.ravel()
		# cv2.circle(img1,(x,y),3,255,-1)
		kp = cv2.KeyPoint()
		kp.pt = (x,y)
		kp.size = 4
		kp1.append(kp)

	kp2 = []
	for i in corners2:
		x,y = i.ravel()
		# cv2.circle(img2,(x,y),3,255,-1)
		kp = cv2.KeyPoint()
		kp.pt = (x,y)
		kp.size = 4
		kp2.append(kp)

	# create descriptors
	kp1, des1 = brief.compute(img1,kp1)
	kp2, des2 = brief.compute(img2,kp2)

	#Match keypoints
	bf = cv2.BFMatcher(crossCheck=True)
	matches = bf.match(des1,des2)

	#sort by distance
	matches = sorted(matches, key = lambda x:x.distance)
	img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,outImg=None ,flags=2)
	# this number determines how many matches to draw ^

	#Show matches
	cv2.imshow('ShiTomasi - ' + description,img3)
	cv2.imwrite('ShiTomasi-' + description + '.jpg',img3)