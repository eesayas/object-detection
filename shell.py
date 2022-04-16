from cmd import Cmd
import os
from collect import collect
from constants import IMAGES_FOLDER
from flagparser import FlagParser
from load import load_pretrained_model
# from detect_realtime import detect_realtime
# from detect_image import detect_image

class Shell(Cmd):
  prompt = '>>> '

  def do_exit(self, inp):
    print('Goodbye!')
    return True

  '''-------------------------------------------------------------------------------------------------
    1. Collect Images
  -------------------------------------------------------------------------------------------------'''
  def do_collect(self, inp):
    flags = FlagParser(inp)

    if not flags.get('limit'):
      print('No limit passed default to 5')
    else:
      if not str(flags.get('limit')).isnumeric():
        print('Passed limit must be a number')
        return

    if not flags.get('folder'):
      print('No folder passed default to {}'.format(IMAGES_FOLDER))
    else:
      if not os.path.exists(flags.get('limit')):
        print('The folder passed does not exists')
        return

    # load defaults
    limit = flags.get('limit') or 5
    folder = flags.get('folder') or IMAGES_FOLDER

    if not flags.get('labels'):
      print('You must provide labels (ex: --labels thumbsup thumbsdown)')
      return

    collect(limit, flags.get('labels'), folder)

  def help_collect(self):
    print('This command will collect images for training and testing')
    print('Example: collect --labels <label1> <label2> --limit <limit> --folder <images_folder>')
    print('limit and folder are optional')

  '''-------------------------------------------------------------------------------------------------
    2. Label Images
  -------------------------------------------------------------------------------------------------'''
  def do_label (self, inp): 
    flags = FlagParser(inp)

    if not flags.get('folder'):
      print('No folder passed default to {}'.format(IMAGES_FOLDER))
    
    folder = flags.get('folder') or IMAGES_FOLDER

    os.system('labelImg {}'.format(folder))
    
  def help_label(self):
    print('This command will open LabelImg to label the collected images')
    print('Example: label --folder <images_folder>')
    print('folder is optional')

  '''-------------------------------------------------------------------------------------------------
    3. (Optional Load pretrained model
  -------------------------------------------------------------------------------------------------'''
  def do_load(self, inp):
    flags = FlagParser(inp)
    if not flags.get('url'):
      print('You must provide the url of the pretained model (ex: --url <pretrained_model_url>')
      return
        
    url = flags.get('url')
    load_pretrained_model(url)

  def help_load(self):
    print('This command will download a pretrained model from Tensorflow 2 Detection Model Zoo')
    print('Example: load --url <pretrained_model_url>')
    print('See: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md')

  '''-------------------------------------------------------------------------------------------------
    4. Train model (flags: --model, --labels, --train )
  -------------------------------------------------------------------------------------------------'''
  def train(self, inp):
    flags = FlagParser(inp)

  '''-------------------------------------------------------------------------------------------------
    5. Detect using TFOD
  -------------------------------------------------------------------------------------------------'''
  # def do_detect(self, inp):
  #   flags = FlagParser(inp)

  #   type = flags.get('type') or 'realtime' # realtime is default

  #   model = flags.get('model')
  #   if model is False:
  #     print('--model is required')
  #     return

  #   if type == 'image':
  #     if not flags.get('image'):
  #       print('You must provide an image to detect (ex: --image path_to_image)')
  #       return

  #     image = flags.get('image')

  #     if not os.path.exists(image):
  #       print('The provide image does not exists')
  #       return

      # detect_image(model, image)

    # elif type == 'realtime':
      # detect_realtime(model)

Shell().cmdloop()