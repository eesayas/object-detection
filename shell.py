from cmd import Cmd
import os
from collect import collect
import cv2

IMAGES_FOLDER = "collectedimages"

def shell():
  while True:
    command = input(">>> ")
    if command == "exit":
      break

    elif command == "collect":
      collect()

shell()