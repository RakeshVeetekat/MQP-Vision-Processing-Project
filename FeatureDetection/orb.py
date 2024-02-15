import cv2
import os


def runOrb(img1,img2, description):

	#create orb dectector
	orb = cv2.ORB_create()

	#find keypoints and descriptors
	kp1, des1 = orb.detectAndCompute(img1,None)
	kp2, des2 = orb.detectAndCompute(img2, None)

	#Match keypoints
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1,des2)

	#sort by distance
	matches = sorted(matches, key = lambda x:x.distance)
	img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,outImg=None ,flags=2)
	# this number determines how many matches to draw ^

	#Show matches
	cv2.imshow('ORB - ' + description,img3)
	path = 'ORB-' + description + '.jpg'
	# print(path)
	cv2.imwrite(path,img3)

