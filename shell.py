from cmd import Cmd
import os
from collect import collect
from constants import IMAGES_FOLDER
from label import label
from flagparser import FlagParser
from setup import setup
from detect_realtime import detect_realtime
from detect_image import detect_image

class Shell(Cmd):
  prompt = '>>> '

  def do_exit(self, inp):
    print('Goodbye!')
    return True

  '''-------------------------------------------------------------------------------------------------
    1. Collect Images

    Usage: collect --limit 5 --labels thumbsup thumbsdown
  -------------------------------------------------------------------------------------------------'''
  def do_collect(self, inp):
    flags = FlagParser(inp)

    if not flags.get('limit'):
      print('No limit passed default to 5')
    
    else:
      if not str(flags.get('limit')).isnumeric():
        print('Passed limit must be a number')
        return
    
    limit = flags.get('limit') or 5 # default limit is 5

    if not flags.get('labels'):
      print('You must provide labels (ex: --labels thumbsup thumbsdown)')
      return

    collect(limit, flags.get('labels'))

  def help_collect(self):
    print('This command will collect images for training and testing')
    print('Example: collect --limit 5 --labels thumbsup thumbsdown')

  '''-------------------------------------------------------------------------------------------------
    2. Label Images

    Usage: label
  -------------------------------------------------------------------------------------------------'''
  def do_label (self, inp):  
    label(IMAGES_FOLDER)
    
  def help_label(self):
    print('This command will open LabelImg to label the collected images')
    print('Example: label')

  '''-------------------------------------------------------------------------------------------------
    3. Setup Tensorflow
  -------------------------------------------------------------------------------------------------'''
  def do_setup(self, inp):
    setup()
  
  def help_setup(self):
    print("Use 'setup' to install Tensorflow and setup other dependencies")
    print("This should only be ran once since it takes a long time to install")

  '''-------------------------------------------------------------------------------------------------
    4. Train model (flags: --model, --labels, --train )
  -------------------------------------------------------------------------------------------------'''
  def train(self, inp):
    flags = FlagParser(inp)
    


  '''-------------------------------------------------------------------------------------------------
    5. Detect using TFOD
  -------------------------------------------------------------------------------------------------'''
  def do_detect(self, inp):
    flags = FlagParser(inp)

    type = flags.get('type') or 'realtime' # realtime is default

    model = flags.get('model')
    if model is False:
      print('--model is required')
      return

    if type == 'image':
      if not flags.get('image'):
        print('You must provide an image to detect (ex: --image path_to_image)')
        return

      image = flags.get('image')

      if not os.path.exists(image):
        print('The provide image does not exists')
        return

      detect_image(model, image)

    elif type == 'realtime':
      detect_realtime(model)

Shell().cmdloop()