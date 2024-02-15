import cv2
import pandas as pd

#location of the folder to save the images
pathToImg = "data\\images\\D12"
#location of the video
pathToVid = "data\\videos\\D12\\flight_video.mp4"


#open file
cap = cv2.VideoCapture(pathToVid)
while not cap.isOpened():
  cap = cv2.VideoCapture(pathToVid)
  cv2.waitKey(1000)
  print ("Wait for the header")


fps = int(cap.get(cv2.CAP_PROP_FPS))

success = True
count = 0

# while there are still images that can be read in video
while success:
  #read image
  success, image = cap.read()
  if count%(2*fps) == 0: # get frame every 2 seconds
    # save image
    imgname = pathToImg + "/img" + str(int(count/(2*fps))) + ".jpg"
    cv2.imwrite(imgname, image)
  count += 1
