import os
from constants import PRETRAINED_MODEL_NAME, PRETRAINED_MODEL_PATH, PRETRAINED_MODEL_URL
import wget
import shutil

def load_pretrained_model(url = PRETRAINED_MODEL_URL, name = PRETRAINED_MODEL_NAME):

    # if pretrained models folder does not exists
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        os.mkdir(PRETRAINED_MODEL_PATH)

    wget.download(url)

    tarfile = '{}.tar.gz'.format(name)

    shutil.unpack_archive(tarfile, PRETRAINED_MODEL_PATH)

    # delete tarfile after unpack
    os.remove(tarfile)
