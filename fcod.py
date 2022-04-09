from cmd import Cmd
import os
import uuid
import cv2

IMAGES_FOLDER = "collectedimages"

class FCODShell(Cmd):
  prompt = "fcod> "

  # Collect images using webcam
  def do_collect(self, inp):
    labels = inp.split()
    index = 0
    counter = 0
    limit = 5

    cap = cv2.VideoCapture(0)

    while(True):
      if index > len(labels):
        break

      ret, frame = cap.read()
      cv2.putText(frame, "press 'c' to capture", (950, 640), cv2.FONT_HERSHEY_SIMPLEX, 1, (57, 252, 3, 255), 3)
      cv2.putText(frame, "press 'q' to quit", (950, 680), cv2.FONT_HERSHEY_SIMPLEX, 1, (57, 252, 3, 255) , 3)
      cv2.imshow('frame', frame)


      pressedKey = cv2.waitKey(1) & 0xFF

      if pressedKey == ord('q'):
        break

      elif pressedKey == ord('c'):
        disp = labels[index] + ' ' + str(counter + 1) + '/' + str(limit)
        print(disp)
        # cv2.putText(frame, disp, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (57, 252, 3, 255), 3)

        counter+=1
        if counter == limit:
          counter = 0
          index+=1
          

    cap.release()
    cv2.destroyAllWindows()

  # Exit the shell
  def do_exit(self, inp):
    print("\nBye")
    return True

  # Fall back if not a command
  def default(self, inp):
    os.system(inp)
 
FCODShell().cmdloop()