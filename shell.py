from cmd import Cmd
import os
from collect import collect
import cv2

class Shell(Cmd):
  prompt = '>>> '

  def do_exit(self, inp):
    print('Goodbye!')
    return True

  def do_collect(self, inp):
    parsed = inp.split()

    if len(parsed) < 2:
      print('Need to provide limit and labels (ex: collect 5 thumbsup thumbsdown)')
    
    elif parsed[0].isnumeric() == False:
      print('First argument must be a number followed by labels (ex: collect 5 thumbsup thumbsdown)')

    else:
      collect(parsed.pop(0), parsed)

  def help_collect(self):
    print("Use 'collect' command to collect images for labelling")
    print('Provide a limit followed by labels (ex: collect 5 thumbsup thumbsdown)')

Shell().cmdloop()