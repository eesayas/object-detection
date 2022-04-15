from cmath import pi
import os
from gitclone import gitclone
from constants import API_MODEL_PATH, LABEL_MAP, PRETRAINED_MODEL_NAME, PRETRAINED_MODEL_URL, PRETRAINED_MODEL_PATH, PROTOC_MAC_URL, PROTOC_MAC_ZIP, PROTOC_PATH, TEST_IMAGES, TEST_TF_RECORD, TF_RECORD_SCRIPT, TRAIN_IMAGES, TRAIN_TF_RECORD
import shutil
import wget
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

'''------------------------------------------------------------------------------
api_model

Description: Download the api model
------------------------------------------------------------------------------'''
def api_model():
    if os.path.exists(API_MODEL_PATH):
        shutil.rmtree(API_MODEL_PATH)
    
    os.mkdir(API_MODEL_PATH)

    gitclone('https://github.com/tensorflow/models', API_MODEL_PATH)

'''------------------------------------------------------------------------------
pretrained_model

Description: Download the pretrained model
Arguments:
- url: the download url
- name: the name of the model
------------------------------------------------------------------------------'''
def pretrained_model(url = PRETRAINED_MODEL_URL, name = PRETRAINED_MODEL_NAME):
    if os.path.exists(PRETRAINED_MODEL_PATH):
        shutil.rmtree(PRETRAINED_MODEL_PATH)

    os.mkdir(PRETRAINED_MODEL_PATH)

    wget.download(url)

    tarfile = '{}.tar.gz'.format(name)

    shutil.unpack_archive(tarfile, PRETRAINED_MODEL_PATH)

    os.remove(tarfile)

'''------------------------------------------------------------------------------
protoc

Description: Download and run protoc
------------------------------------------------------------------------------'''
def protoc():
    if os.path.exists(PROTOC_PATH):
        shutil.rmtree(PROTOC_PATH)
    
    os.mkdir(PROTOC_PATH)

    if os.name == 'posix':
        # os.environ['PATH'] = os.path.join(os.getcwd(), 'protoc', 'bin') + ':' + os.environ['PATH']
        # os.system('protoc --version')

        # download protoc for mac
        wget.download(PROTOC_MAC_URL)
        shutil.unpack_archive(PROTOC_MAC_ZIP, PROTOC_PATH)
        os.remove(PROTOC_MAC_ZIP)

        # permissions 
        os.system("chmod +x protoc/bin/protoc")

        # add protoc to path
        os.environ['PATH'] = os.path.join(os.getcwd(), 'protoc', 'bin') + ':' + os.environ['PATH']

        # run
        os.system('cd {}/models/research && protoc object_detection/protos/*.proto --python_out=.'.format(os.getcwd()))

'''------------------------------------------------------------------------------
installTF

Description: Install tensorflow and its dependencies
------------------------------------------------------------------------------'''
def installTF():
    if not os.path.exists('models'):
        raise Exception('Models from Tensorflow Garden must be downloaded first')

    currentdir = os.get_cwd()

    if os.name == 'posix':
        os.chdir(os.path.join(currentdir, 'models', 'research'))
        shutil.copy(os.path.join('object_detection', 'packages', 'tf2', 'setup.py'), '.')
        os.system('python -m pip install .')

    if os.name == 'nt':
        print('Windows implentation to be done')    

    os.chdir(currentdir)

'''------------------------------------------------------------------------------
verify

Description: Verify the install of Tensorflow
------------------------------------------------------------------------------'''
def verify():
    verification_script = os.path.join(API_MODEL_PATH, 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
    os.system('python {}'.format(verification_script))


'''------------------------------------------------------------------------------
create_label_map

Description: Create label map file according to label names
Arguments:
- label_names: a list of strings that are case sensitive to labels from LabelImg
------------------------------------------------------------------------------'''
def create_label_map(label_names):
    label_map = []
    index = 1
    for name in label_names:
        label = {}
        label['name'] = name
        label['id'] = index
        label_map.append(label)

    with open('annotations/label_map.pbtxt', 'w') as f:
        for label in label_map:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')

'''------------------------------------------------------------------------------
create_tf_records

Description: Create TF records
------------------------------------------------------------------------------'''
def create_tf_records():

    os.system('python {} -x {} -l {} -o {}'.format(TF_RECORD_SCRIPT, TRAIN_IMAGES, LABEL_MAP, TRAIN_TF_RECORD))
    os.system('python {} -x {} -l {} -o {}'.format(TF_RECORD_SCRIPT, TEST_IMAGES, LABEL_MAP, TEST_TF_RECORD))


'''------------------------------------------------------------------------------
update_pipeline_config

Description: Update Config for Transfer Learning
Arguments:
- number_of_labels:
- model_name: 
------------------------------------------------------------------------------'''
def update_pipeline_config(number_of_labels, model_name):

    # Copy Model Config to Training Folder
    pipeline_config_path = os.path.join(PRETRAINED_MODEL_PATH, PRETRAINED_MODEL_NAME, 'pipeline.config')
    custom_model_path = os.path.join(API_MODEL_PATH, model_name)
    shutil.copy(pipeline_config_path, custom_model_path)
    
    # Update Config for Transfer Learning
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(pipeline_config_path, 'r') as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)
    
    pipeline_config.model.ssd.num_classes = number_of_labels
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine.tune_checkpoint = os.path.join(PRETRAINED_MODEL_PATH, PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
    pipeline_config.train_config.fine_tune_checkpoint_type = 'detection'
    pipeline_config.train_input_reader.label_map_path = LABEL_MAP
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [TRAIN_TF_RECORD]
    pipeline_config.eval_input_reader[0].label_map_path = LABEL_MAP
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [TEST_TF_RECORD]

    # Rewrite pipeline config
    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(os.path.join(custom_model_path, 'pipeline.config'), 'wb') as f:
        f.write(config_text)

'''------------------------------------------------------------------------------
setup

Description: Aggregate all functions above
Arguments:
- label: list of labels
- model_name: custom name for model to be trained
------------------------------------------------------------------------------'''
def setup(labels, model_name):
    # Download TF Models Pretrained Models from Tensorflow Model Zoo and Install TFOD
    # api_model()
    pretrained_model()
    
    # Install and run protoc
    # protoc()
    # installTF()

    # Verify Tensorflow
    # verify()

    # Create Label Map
    # create_label_map(labels)
    
    # Create TF records
    # create_tf_records()

    # Update Config For Transfer Learning
    # update_pipeline_config(len(labels), model_name)


setup(['ThumbsUp', 'ThumbsDown'], 'mymodel')
