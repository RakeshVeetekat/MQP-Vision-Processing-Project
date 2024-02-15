import cv2


def runSift(img1,img2, description):

	#create sift dectector
	sift = cv2.xfeatures2d.SIFT_create()

	#find keypoints and descriptors
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2, None)

	#Match keypoints
	bf = cv2.BFMatcher(crossCheck=True)
	matches = bf.match(des1,des2)

	#sort by distance
	matches = sorted(matches, key = lambda x:x.distance)
	img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,outImg=None ,flags=2)
	# this number determines how many matches to draw ^

	#Show matches
	cv2.imshow('Sift - ' + description,img3)
	cv2.imwrite('Sift-' + description + '.jpg',img3)