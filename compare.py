import cv2
from orb import runOrb
from shitomasi import runShitomasi
from sift import runSift
import os

orb = True
shitomasi = True
sift = True

def runAlgos(img1, img2, description):
	if(orb):
		runOrb(img1, img2, description)
	if(shitomasi):
		runShitomasi(img1, img2, description)
	if(sift):
		runSift(img1, img2, description)

def next():
	k = cv2.waitKey(0)
	# esc key
	if k == 27:
		exit(0)
	cv2.destroyAllWindows()



img1 = cv2.imread('img1652.png',0)
img2 = cv2.imread('img1658.png',0)

runAlgos(img1, img2, 'ensemble')
next()

# img1 = cv2.imread('img1652.png',1)
# img2 = cv2.imread('img1652GE.png',1)
# img3 = cv2.imread('img1658.png',1)
# img4 = cv2.imread('img1652GE-preprocessed.png',1)
# img5 = cv2.imread('img1658-preprocessed.png',1)

# runAlgos(img1, img2, 'GE no preprocessing')
# next()
# runAlgos(img1, img4, 'GE with preprocessing')
# next()
# runAlgos(img1, img3, 'Video no preprocessing')
# next()
# runAlgos(img1, img5, 'Video with preprocessing')
# next()