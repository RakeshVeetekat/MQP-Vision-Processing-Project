import numpy as np
import pandas as pd
import cv2
from progress.bar import IncrementalBar

VidToVid = 0
VidToGE = 0
GEToGE = 0

orb = 0
shitomasi = 0
sift = 0

def orbDetector():
  orb = cv2.ORB_create()

  kp1 = orb.detect(img1, None)
  kp2 = orb.detect(img2, None)
  return kp1,kp2


def shitomasiDetector():
  corners1 = cv2.goodFeaturesToTrack(img1,1000,0.01,10)
  corners1 = np.int0(corners1)

  corners2 = cv2.goodFeaturesToTrack(img2,1000,0.01,10)
  corners2 = np.int0(corners2)

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
  return kp1,kp2


def siftDetector():
  sift = cv2.SIFT_create()

  kp1 = sift.detect(img1,None)
  kp2 = sift.detect(img2, None)
  return kp1,kp2

serialNum1 = input("What is the serial number for the first image? ")
serialNum2 = input("What is the serial number for the second image? ")
imageType = input("0 for Video and Video, 1 for Video and GE, 2 " + 
	"for GE and GE. ")
algo = input("0 for ORB, 1 for Shi-Tomasi, 2 for SIFT. ")

if imageType == '0':
  VidToVid = 1
elif imageType == '1':
  VidToGE = 1
elif imageType == '2':
  GEToGE = 1

if algo == '0':
  orb = 1
elif algo == '1':
  shitomasi = 1
elif algo == '2':
  sift = 1

data = []
data.append(['Serial Number 1','Serial Number 2', 'Match Number', 
	'ORB','SIFT','Shi-Tomasi','Vid and Vid','Vid and GE',
	'GE and GE','Distance','Descriptor 1',
	'Descriptor 2','Correctness'])

# read images
img1 = cv2.imread(serialNum1 + '.jpg',0)
img2 = cv2.imread(serialNum2 + '.jpg',0)


#equalize histogram
img1 = cv2.equalizeHist(img1)
img2 = cv2.equalizeHist(img2)


#get key points
kp1 = []
kp2 = []
if algo == '0':
  kp1,kp2 = orbDetector()
elif algo == '1':
  kp1,kp2 = shitomasiDetector()
elif algo == '2':
  kp1,kp2 = siftDetector()

#get descriptors
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
kp1, des1 = brief.compute(img1,kp1)
kp2, des2 = brief.compute(img2,kp2)

#match keypoints
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)


#loop through matches
bar = IncrementalBar('Amount of matches', max = len(matches))
for i in range(0,len(matches)):

  # Draw a single match
  img3 = cv2.drawMatches(img1,kp1,img2,kp2,[matches[i]],
  	matchColor = (0,0,255),outImg=None ,flags=2)

  # Store details of match
  distance = matches[i].distance
  matcherObject = matches[i]
  descriptor2 = des2[matches[i].trainIdx]
  descriptor1 = des1[matches[i].queryIdx]

  # turn descriptors into a bitstring
  desbits1 = ''
  for t in descriptor1:
    desbits1 += ('{:08b}'.format(t))

  desbits2 = ''
  for t in descriptor2:
    desbits2 += ('{:08b}'.format(t))

  cv2.imshow('matches', img3)
  k = cv2.waitKey(0)
  # if incorrect match, press n key
  # Otherwise, any other key press is assumed as a correct match
  correctMatch = 1
  # n key, incorrect match (-1)
  if k == 110:
    correctMatch = -1
  # if m key, not sure (0)
  if k == 109:
    correctMatch = 0
  # esc key, use to generate csv file
  if k == 27:
    df = pd.DataFrame(data)
    df.to_csv("foo.csv")
    break
  # add data to matrix
  bar.next()
  data.append([serialNum1, serialNum2, i, orb, sift, shitomasi,
   VidToVid, VidToGE, GEToGE, distance, desbits1, desbits2,
   correctMatch])

# add data to csv
df = pd.DataFrame(data)
df.to_csv("foo.csv")



