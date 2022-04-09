import numpy as np
import cv2

def collect():
  cap = cv2.VideoCapture(0)
  
  while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    keypressed = cv2.waitKey(1)
    if keypressed == ord('c'):
      print('capture!')
    elif keypressed == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()
  cv2.waitKey(1)

