import cv2
# import pyperclip

# read images
img1 = cv2.imread('point500.png',0)
img2 = cv2.imread('5.jpg',0)

#create sift
sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2, None)


# create BFMatcher object
bf = cv2.BFMatcher()

# Match descriptors.
matches = bf.match(des1,des2)

#sort matches
matches = sorted(matches, key = lambda x:x.distance)


for i in range(0,1000):
	# draw matches
	img3 = cv2.drawMatches(img1,kp1,img2,kp2,[matches[i]],matchColor = (0,255,0), outImg=None ,flags=2)

	string = '#' + str(i) + ':  ' + str(matches[i].distance)
	#print string
	print(string)
	#copy string
	# pyperclip.copy(string)

	# show matches
	cv2.imshow('matches',img3)
	k = cv2.waitKey(0)
	# esc key
	if k == 27:
		break

