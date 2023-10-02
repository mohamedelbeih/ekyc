# import modules
import cv2
import time


# open file
cap = cv2.VideoCapture('video.avi')

# get FPS of input video
fps = cap.get(cv2.CAP_PROP_FPS)
breakpoint()
count = 0
# read and write frams for output video
while cap.isOpened():
	ret, frame = cap.read()
	if not ret:
		break
        
	cv2.imwrite("test/{}.jpg".format(count),frame)
	count += 1
	

# release resources
cap.release()


