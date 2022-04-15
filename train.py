import os
from constants import API_MODEL_PATH
from gitclone import gitclone
from create_label_map import create_label_map

def train(labels):
    # Create annotations folder
    if not os.path.exists('annotations'):
        os.mkdir('annotations')

    # Create label map
    create_label_map(labels)
    

train(['ThumbsDown', 'ThumbsUp'])