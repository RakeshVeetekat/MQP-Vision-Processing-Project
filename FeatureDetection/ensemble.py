import cv2
import numpy as np


#distance limit for each algorithm
orb_limit = 50
sift_limit = 120
shitomasi_limit = 1

#arrays of the good keypoints and matches
good_kp1 = []
good_des1 = []
good_kp2 = []
good_des2 = []
good_matches = []
#keeps track of how many matches have been added
global amount
amount = 0

def runOrb():
	global amount
	orb = cv2.ORB_create()
	#find keypoints and descriptors
	kp1, des1 = orb.detectAndCompute(img1,None)
	kp2, des2 = orb.detectAndCompute(img2, None)
	#Match keypoints
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1,des2)

	for mat in matches:
		if(mat.distance < orb_limit):
			#add good keypoints and descriptors
			good_kp1.append(kp1[mat.queryIdx])
			good_des1.append(des1[mat.queryIdx])
			good_kp2.append(kp2[mat.trainIdx])
			good_des2.append(des2[mat.trainIdx])

			# make and add new match
			match = cv2.DMatch(amount, amount, mat.distance)
			good_matches.append(match)

			amount += 1
			print('added orb')


def runSift():
	global amount
	sift = cv2.xfeatures2d.SIFT_create()

	#find keypoints and descriptors
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2, None)

	#Match keypoints
	bf = cv2.BFMatcher(crossCheck=True)
	matches = bf.match(des1,des2)

	for mat in matches:
		if(mat.distance < sift_limit):
			#add good keypoints and descriptors
			good_kp1.append(kp1[mat.queryIdx])
			good_des1.append(des1[mat.queryIdx])
			good_kp2.append(kp2[mat.trainIdx])
			good_des2.append(des2[mat.trainIdx])

			# make and add new match
			match = cv2.DMatch(amount, amount, mat.distance)
			good_matches.append(match)

			amount += 1
			print('added sift')


def runShiTomasi():
	global amount
	global img1
	global img2
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

	for mat in matches:
		if(mat.distance < shitomasi_limit):
			#add good keypoints and descriptors
			good_kp1.append(kp1[mat.queryIdx])
			good_des1.append(des1[mat.queryIdx])
			good_kp2.append(kp2[mat.trainIdx])
			good_des2.append(des2[mat.trainIdx])

			# make and add new match
			match = cv2.DMatch(amount, amount, mat.distance)
			good_matches.append(match)

			amount += 1
			print('added shitomasi')

#get two images
img1 = cv2.imread('img1652GE-preprocessed.png',0)
img2 = cv2.imread('img1658-preprocessed.png',0)

#run all three algorithms
runOrb()
print(amount)
runSift()
print(amount)
runShiTomasi()
print(amount)
print()

#display matches
img3 = cv2.drawMatches(img1,good_kp1,img2,good_kp2,good_matches,outImg=None ,flags=2)
cv2.imshow('ensemble', img3)
k = cv2.waitKey(0)


cv2.imwrite('ensemble.png', img3)

