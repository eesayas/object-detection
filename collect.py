import shutil
from tkinter.tix import IMAGE
import numpy as np
import cv2
from constants import IMAGES_FOLDER
import uuid
import os

def config_title(label, current, limit):
  return "Collecting {}  ({} of {}). Press 'c' to capture.".format(label, current, limit)

def collect(limit, labels):

  # Setup folders
  if not os.path.exists(IMAGES_FOLDER):
    os.mkdir(IMAGES_FOLDER)

  for label in labels:
    path = os.path.join(IMAGES_FOLDER, label)
    if not os.path.exists(path):
      os.mkdir(path)

  limit = int(limit)
  cap = cv2.VideoCapture(0)
  print('Opening camera...')
  
  current = 1
  labelIndex = 0
  title = config_title(labels[labelIndex], current, limit)

  while True:
    ret, frame = cap.read()
    cv2.imshow(title, frame)

    keypressed = cv2.waitKey(1)
    if keypressed == ord('c'):

      # prevent labelIndex exceeding labels array
      if labelIndex == len(labels):
        continue

      # capture current image
      imgname = os.path.join(IMAGES_FOLDER, labels[labelIndex], labels[labelIndex] + '.' + '{}.jpg'.format(str(uuid.uuid1())))
      cv2.imwrite(imgname, frame)

      # queue next image
      current += 1

      # if next image is over the limit do a reset and go the next label
      if current > limit:
        current = 1 # reset
        labelIndex += 1 # next label
      
      # if all labels are discovered
      if labelIndex == len(labels):
        title = "All labels are captured. Please press 'q' to quit"
      
      else:
        title = config_title(labels[labelIndex], current, limit)

    elif keypressed == ord('q'):
      break

  print('Closing camera')
  print('Images saved in {}'.format(os.path.join(os.getcwd(), IMAGES_FOLDER)))

  cap.release()
  cv2.destroyAllWindows()
  cv2.waitKey(1)

