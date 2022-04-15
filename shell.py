from cmd import Cmd
import os
from collect import collect
import cv2
from label import label
from flagparser import flagparser
from partition import partition
from setup import setup
from constants import IMAGES_FOLDER

class Shell(Cmd):
  prompt = '>>> '

  def do_exit(self, inp):
    print('Goodbye!')
    return True

  '''-------------------------------------------------------------------------------------------------
    Collect Images
  -------------------------------------------------------------------------------------------------'''
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

  
  '''-------------------------------------------------------------------------------------------------
    Label Images
  -------------------------------------------------------------------------------------------------'''
  def do_label (self, inp):
    if len(inp.split()) > 1:
      print('Only 1 argument is permitted which is the folder of images (ex: label collectedimages)')
    
    elif len(inp) == 0:
      label(False)
    
    else:
      if not os.path.exists(inp):
        print('Image folder ' + "'" + inp + "'" + ' does not exists')
        
      else:
        label(inp)
    
  def help_label(self):
    print("Use 'label' command to label images for training")
    print("You may specify a folder of images to label or not (default to 'collectedimages' folder)")
    print("(ex: 'label' or 'label myimagefolder')")

  '''-------------------------------------------------------------------------------------------------
    Partition Images for training and testing sets
  -------------------------------------------------------------------------------------------------'''
  def do_partition(self, inp):
    flags = flagparser(inp)
    
    if('train' not in flags or flags['train'] is False):
      print('--train flag and value is required to run this command')
      return
    
    if('folder' not in flags or flags['folder'] is False):
      flags['folder'] = IMAGES_FOLDER

    partition(flags)

  '''-------------------------------------------------------------------------------------------------
    Setup Tensorflow
  -------------------------------------------------------------------------------------------------'''
  def do_setup(self, inp):
    setup()
  
  def help_setup(self):
    print("Use 'setup' to install Tensorflow and setup other dependencies")
    print("This should only be ran once since it takes a long time to install")

Shell().cmdloop()