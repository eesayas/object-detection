import os
from constants import PRETRAINED_MODEL_NAME, PRETRAINED_MODEL_PATH, PRETRAINED_MODEL_URL
import wget
import shutil

'''------------------------------------------------------------------------------
load_pretrained_model

Description: Download a pretrained model
Arguments:
- url: link to model
------------------------------------------------------------------------------'''
def load_pretrained_model(url = PRETRAINED_MODEL_URL):

    # if pretrained models folder does not exists
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        os.mkdir(PRETRAINED_MODEL_PATH)

    wget.download(url)

    tarfile = url.split('/')[-1]

    shutil.unpack_archive(tarfile, PRETRAINED_MODEL_PATH)

    # delete tarfile after unpack
    os.remove(tarfile)

    print('\nSuccessfully downloaded {}'.format(tarfile))
