# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
# import sys
# sys.path.append(os.path.abspath('..')) 

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting face capture video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()

name = input("Name of person:")
dirPath = os.getcwd() + "/facecapture/" + name

if not os.path.isdir(dirPath):
	print('The directory is not present. Creating a new one..')
	os.mkdir(dirPath)
else: 
	print('Directory exists')

# loop over frames from the video file stream
cnt = 0
while cnt < 300: # 300 = captures 30 images
	# grab the frame from the threaded video stream
	frame = vs.read()
	
	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
	cv2.resize(frame, (300, 300)), 1.0, (300, 300),
	(104.0, 177.0, 123.0), swapRB=False, crop=False)

	protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
	modelPath = os.path.sep.join(["face_detection_model",
		"res10_300x300_ssd_iter_140000.caffemodel"])
	detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for the
			# face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI
			face = frame[startY:endY, startX:endX]

		if i == 0 and cnt % 10 == 0:
			filepath = name + "_" + str(cnt) + "_" + str(i) + ".jpg"
			fullpath = os.path.join(os.getcwd(),"facecapture",name,filepath)

			cv2.imwrite(fullpath, face)
			print("Saved: ", fullpath)

	# update the FPS counter
	fps.update()

    # show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	cnt += 1

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()