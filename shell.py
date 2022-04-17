from cmd import Cmd
import os
from collect import collect
from constants import IMAGES_FOLDER
from flagparser import FlagParser
from load import load_pretrained_model
from train import train
from detect_realtime import detect_realtime

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
  def do_train(self, inp):
    flags = FlagParser(inp)

    # Steps
    if not flags.get('steps'):
      print('No steps provided default to 2000')
    
    steps = flags.get('steps') or 2000

    # Pretrained
    if not flags.get('pretrained'):
      print('No pretrained model provided default to ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8')

    pretrained = flags.get('pretrained') or 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'

    # Folder
    if not flags.get('folder'):
      print('No image folder provided default to {}'.format(IMAGES_FOLDER))

    folder = flags.get('folder') or IMAGES_FOLDER

    # Model
    if not flags.get('model'):
      print('You must provide a name for your model (ex: --model <model_name>)')
      return

    model = flags.get('model')

    # Labels
    if not flags.get('labels'):
      print('You must provide labels (ex: --labels <label1> <label2>)')
      return

    labels = flags.get('labels')

    # Sample
    if not flags.get('sample'):
      print('You must indicate number of images to be trained (ex: --train 3)')
      return

    sample = flags.get('sample')

    train(model, labels, folder, sample, pretrained, steps)

  '''-------------------------------------------------------------------------------------------------
    5. Test a trained model
  -------------------------------------------------------------------------------------------------'''
  def do_test(self, inp):
    flags = FlagParser(inp)

    model = flags.get('model')

    if not model:
      print('You must provide a model (ex: --model <model_name>)')
      return
    
    print('Opening webcam. Please wait...')
    detect_realtime(model)

Shell().cmdloop()