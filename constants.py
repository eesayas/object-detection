import os

IMAGES_FOLDER = "collectedimages"
LABELIMG_PATH = "labelimg"

PRETRAINED_MODEL_PATH = 'pretrained-models'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'

API_MODEL_PATH = 'models'

PROTOC_PATH = 'protoc'
PROTOC_MAC_ZIP = 'protoc-3.20.0-osx-x86_64.zip'
PROTOC_MAC_URL = 'https://github.com/protocolbuffers/protobuf/releases/download/v3.20.0/protoc-3.20.0-osx-x86_64.zip'

ANNOTATIONS_PATH = 'annotations'
TF_RECORD_SCRIPT = 'generate_tfrecord.py'

TRAIN_IMAGES= 'train'
TEST_IMAGES = 'test'
LABEL_MAP = os.path.join(ANNOTATIONS_PATH, 'label_map.pbtxt')
TRAIN_TF_RECORD = os.path.join(ANNOTATIONS_PATH, 'train.record')
TEST_TF_RECORD = os.path.join(ANNOTATIONS_PATH, 'test.record')