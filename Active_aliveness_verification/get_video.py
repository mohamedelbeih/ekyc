import cv2
import os
import sys
import random

image_folder = 'actions/' + sys.argv[1]
video_name = 'Input/video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images = random.choices(images,k=4)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

video.release()
