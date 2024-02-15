import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfc
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2

tf.compat.v1.disable_eager_execution()

# load pictures
img1 = cv2.imread('1.jpg',0)
img2 = cv2.imread('3.jpg',0)
# determine the kind of image pair
VidToVid = 0
VidToGE = 1
GEToGE = 0

# Normalize histogram
img1 = cv2.equalizeHist(img1)
img2 = cv2.equalizeHist(img2)

# variables to keep track of matches
orb = 0
sift = 0
shitomasi = 0
all_kp1 = []
all_kp2 = []
all_des1 = []
all_des2 = []
all_matches = []
amount = 0 # keeps track of amount of matches
nnData = []

# fuction to put value in the right format for the NN
def getNNData(des1, des2, match):
  #append data needed
  data = [orb, sift, shitomasi, VidToVid, VidToGE, GEToGE,
   match.distance]

  # append descriptors
  des1_bits = ''
  for des in des1:
    des1_bits += ('{:08b}'.format(des))
  des1_bits = list(des1_bits)

  for bit in des1_bits:
    data.append(bit)

  des2_bits = ''
  for des in des2:
    des2_bits += ('{:08b}'.format(des))
  des2_bits = list(des2_bits)

  for bit in des2_bits:
    data.append(bit)

  nnData.append(data)


# get matches with orb
orb = 1
orbDetector = cv2.ORB_create()
#find keypoints and descriptors
kp1, des1 = orbDetector.detectAndCompute(img1,None)
kp2, des2 = orbDetector.detectAndCompute(img2, None)
#Match keypoints
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)

# add key points and descriptors found with ORB
# to all keypoints array
for mat in matches:
  all_kp1.append(kp1[mat.queryIdx])
  all_des1.append(des1[mat.queryIdx])
  all_kp2.append(kp2[mat.trainIdx])
  all_des2.append(des2[mat.trainIdx])

  # make and add new match
  match = cv2.DMatch(amount, amount, mat.distance)
  all_matches.append(match)

  amount += 1

  getNNData(des1[mat.queryIdx], des2[mat.trainIdx], match)


# get matches with sift
orb = 0
sift = 1

siftDetector = cv2.xfeatures2d.SIFT_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

#find keypoints 
kp1 = siftDetector.detect(img1,None)
kp2 = siftDetector.detect(img2, None)
# create descriptors
kp1, des1 = brief.compute(img1,kp1)
kp2, des2 = brief.compute(img2,kp2)

#Match keypoints
bf = cv2.BFMatcher(crossCheck=True)
matches = bf.match(des1,des2)

# add key points and descriptors found with SIFT
# to all keypoints array
for mat in matches:
  all_kp1.append(kp1[mat.queryIdx])
  all_des1.append(des1[mat.queryIdx])
  all_kp2.append(kp2[mat.trainIdx])
  all_des2.append(des2[mat.trainIdx])

  # make and add new match
  match = cv2.DMatch(amount, amount, mat.distance)
  all_matches.append(match)

  amount += 1

  getNNData(des1[mat.queryIdx], des2[mat.trainIdx], match)


# get matches with shitomasi
sift = 0
shitomasi = 1

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

# add key points and descriptors found with Shi Tomasi
# to all keypoints array
for mat in matches:
  all_kp1.append(kp1[mat.queryIdx])
  all_des1.append(des1[mat.queryIdx])
  all_kp2.append(kp2[mat.trainIdx])
  all_des2.append(des2[mat.trainIdx])

  # make and add new match
  match = cv2.DMatch(amount, amount, mat.distance)
  all_matches.append(match)

  amount += 1

  getNNData(des1[mat.queryIdx], des2[mat.trainIdx], match)


# initialize variables for NN 
training_epochs = 200

n_neurons_in_h1 = 500
n_neurons_in_h2 = 250
n_neurons_in_h3 = 100
n_neurons_in_h4 = 50
n_neurons_in_h5 = 10
learning_rate = 0.0001

n_features = 519
labels_dim = 1
#############################################

# these placeholders serve as our input tensors
x = tfc.placeholder(tf.float32, [None, n_features], name='input')
y = tfc.placeholder(tf.float32, [None, labels_dim], name='labels')

#weights and biases for layer 1
W1 = tf.Variable(tfc.truncated_normal([n_features, n_neurons_in_h1],
 mean=0, stddev=1 / np.sqrt(n_features)),name='weights1')
b1 = tf.Variable(tfc.truncated_normal([n_neurons_in_h1],
 mean=0, stddev=1 / np.sqrt(n_features)), name='biases1')
#out put of layer 1
y1 = tf.nn.leaky_relu((tf.matmul(x, W1) + b1),
 name='activationLayer1')

#layer 2
W2 = tf.Variable(tfc.random_normal([n_neurons_in_h1, n_neurons_in_h2],
 mean=0, stddev=1),name='weights2')
b2 = tf.Variable(tfc.random_normal([n_neurons_in_h2],
 mean=0, stddev=1), name='biases2')
y2 = tf.nn.leaky_relu((tf.matmul(y1, W2) + b2),
 name='activationLayer2')

#layer 3
W3 = tf.Variable(tfc.random_normal([n_neurons_in_h2, n_neurons_in_h3],
 mean=0, stddev=1),name='weights3')
b3 = tf.Variable(tfc.random_normal([n_neurons_in_h3],
 mean=0, stddev=1), name='biases3')
y3 = tf.nn.leaky_relu((tf.matmul(y2, W3) + b3),
 name='activationLayer3')

#layer 4
W4 = tf.Variable(tfc.random_normal([n_neurons_in_h3, n_neurons_in_h4],
 mean=0, stddev=1),name='weights4')
b4 = tf.Variable(tfc.random_normal([n_neurons_in_h4],
 mean=0, stddev=1), name='biases4')
y4 = tf.nn.leaky_relu((tf.matmul(y3, W4) + b4),
 name='activationLayer4')

#layer 5
W5 = tf.Variable(tfc.random_normal([n_neurons_in_h4, n_neurons_in_h5],
 mean=0, stddev=1),name='weights5')
b5 = tf.Variable(tfc.random_normal([n_neurons_in_h5],
 mean=0, stddev=1), name='biases5')
y5 = tf.nn.leaky_relu((tf.matmul(y4, W5) + b5),
 name='activationLayer5')

# output layer weights and biases
Wo = tf.Variable(tfc.random_normal([n_neurons_in_h5, labels_dim],
 mean=0, stddev=1 ),name='weightsOut')
bo = tf.Variable(tfc.random_normal([labels_dim],
 mean=0, stddev=1), name='biasesOut')

# the sigmoid (binary softmax) activation is absorbed into TF's
# sigmoid_cross_entropy_with_logits loss
logits = (tf.matmul(y5, Wo) + bo)

# tap a separate output that applies softmax 
# activation to the output layer
# for training accuracy readout
a = tf.nn.sigmoid(logits, name='activationOutputLayer')

# initialize saver
saver = tfc.train.Saver(var_list = None)

# predict values with neural network
with tfc.Session() as sess:
  # get saved NN
  saver.restore(sess, 'trainedNN')

  Y_pred = sess.run(a, feed_dict={input_X: nnData})
  # predict y values
  print(Y_pred)


good_kp1 = []
good_kp2 = []
good_matches = []
amount = 0
for i in range(0,len(Y_pred)):
  # if probability is greater than .5
  # it is a good match
  if Y_pred[i] > .5:
    match = cv2.DMatch(amount, amount, all_matches[i].distance)
    good_matches.append(match)
    good_kp1.append(all_kp1[all_matches[i].queryIdx])
    good_kp2.append(all_kp2[all_matches[i].trainIdx])

    amount += 1

# draw good matches
img3 = cv2.drawMatches(img1,good_kp1,img2,good_kp2,good_matches,
	outImg=None ,flags=2)
cv2.imshow('ensemble with NN', img3)
k = cv2.waitKey(0)
# save image
cv2.imwrite('ensemble_NN.png', img3)