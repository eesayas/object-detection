import os
import shutil
from constants import ANNOTATIONS_PATH, API_MODEL_PATH, LABEL_MAP, PRETRAINED_MODEL_NAME, PRETRAINED_MODEL_PATH, TEST_IMAGES, TEST_TF_RECORD, TF_RECORD_SCRIPT, TRAIN_IMAGES, TRAIN_TF_RECORD, TRAINING_SCRIPT
import tensorflow as tf
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


'''------------------------------------------------------------------------------
create_label_map

Description: Create label map file according to label names
Arguments:
- label_names: a list of strings that are case sensitive to labels from LabelImg
------------------------------------------------------------------------------'''
def create_label_map(label_names):
    if os.path.exists(ANNOTATIONS_PATH):
        shutil.rmtree(ANNOTATIONS_PATH)
    
    os.mkdir(ANNOTATIONS_PATH)

    print('Creating label map for {}...'.format((', ').join(label_names)))

    label_map = []
    index = 1
    for name in label_names:
        label = {}
        label['name'] = name
        label['id'] = index
        label_map.append(label)
        index+=1

    with open(os.path.join(ANNOTATIONS_PATH, 'label_map.pbtxt'), 'w') as f:
        for label in label_map:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')
        
    print('Successfully created label map')

'''------------------------------------------------------------------------------
create_tf_records

Description: Create TF records
------------------------------------------------------------------------------'''
def create_tf_records():

    print('Creating TF record for training...')
    os.system('python {} -x {} -l {} -o {}'.format(TF_RECORD_SCRIPT, TRAIN_IMAGES, LABEL_MAP, TRAIN_TF_RECORD))
    
    print('Creating TF record for testing...')
    os.system('python {} -x {} -l {} -o {}'.format(TF_RECORD_SCRIPT, TEST_IMAGES, LABEL_MAP, TEST_TF_RECORD))


'''------------------------------------------------------------------------------
update_pipeline_config

Description: Update Config for Transfer Learning
Arguments:
- number_of_labels:
- model_name: 
------------------------------------------------------------------------------'''
def update_pipeline_config(number_of_labels, model_name):
    print('Updating pipeline config...')

    # Copy Model Config to Training Folder
    pipeline_config_path = os.path.join(PRETRAINED_MODEL_PATH, PRETRAINED_MODEL_NAME, 'pipeline.config')
    custom_model_path = os.path.join(API_MODEL_PATH, model_name)
    
    if os.path.exists(custom_model_path):
        shutil.rmtree(custom_model_path)

    os.mkdir(custom_model_path)

    shutil.copy(pipeline_config_path, custom_model_path)
    
    # Update Config for Transfer Learning
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(os.path.join(custom_model_path, 'pipeline.config'), 'r') as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)
    
    pipeline_config.model.ssd.num_classes = number_of_labels
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(PRETRAINED_MODEL_PATH, PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
    pipeline_config.train_config.fine_tune_checkpoint_type = 'detection'
    pipeline_config.train_input_reader.label_map_path = LABEL_MAP
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [TRAIN_TF_RECORD]
    pipeline_config.eval_input_reader[0].label_map_path = LABEL_MAP
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [TEST_TF_RECORD]

    # Rewrite pipeline config
    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(os.path.join(custom_model_path, 'pipeline.config'), 'wb') as f:
        f.write(config_text)

    return custom_model_path


def train(labels, model_name):
    # Create Label Map
    # create_label_map(labels)

    # Create TF Records
    # create_tf_records()

    # Update Config for Transfer Learning
    custom_model_path = update_pipeline_config(len(labels), model_name)

    # Train
    NUM_TRAIN_STEPS = 2000
    os.system('python {} --model_dir={} --pipeline_config_path={} --num_train_steps={}'.format(TRAINING_SCRIPT, custom_model_path, os.path.join(custom_model_path, 'pipeline.config'), NUM_TRAIN_STEPS))

train(['ThumbsDown', 'ThumbsUp'], 'my-model')