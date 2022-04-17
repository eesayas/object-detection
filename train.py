import os
import shutil
from constants import ANNOTATIONS_PATH, API_MODEL_PATH, IMAGES_FOLDER, LABEL_MAP, PRETRAINED_MODEL_NAME, PRETRAINED_MODEL_PATH, TEST_IMAGES, TEST_TF_RECORD, TF_RECORD_SCRIPT, TRAIN_IMAGES, TRAIN_TF_RECORD, TRAINING_SCRIPT
import tensorflow as tf
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

'''--------------------------------------------------------
check_labeling

Description: check if all jpgs are labeled
Args:
- jpgs: dictonary of sets ( ex: { label: { jpg1, jpg2 } } )
- xmls: dictonary of sets ( ex: { label: { xml1, xml2 } } )
--------------------------------------------------------'''
def check_labeling(jpgs, xmls):
    unlabeled = []

    for label in jpgs:
        for jpg in jpgs[label].difference(xmls[label]):
            unlabeled.append(jpg)
    
    return unlabeled

'''------------------------------------------------------------------------
partition

Description: Partition to train and test samples from collectedimages
Arguments:

- folder: image folder containg collected images
- number_of_trainees: population of sample of training set for each label
-------------------------------------------------------------------------'''
def partition(folder, trainees):
    jpgs = {}
    xmls = {}
    index = 0

    for subdir, dirs, files in os.walk(folder):
        for file in files:
            file_name, file_extension = os.path.splitext(file)
            if file_extension == '.jpg':
                # initialize set if not existing
                if subdir not in jpgs:
                    jpgs[subdir] = set([])

                # add jpg
                jpgs[subdir].add(os.path.join(subdir, file_name))

            elif file_extension == '.xml':
                # initialize set if not existing
                if subdir not in xmls:
                    xmls[subdir] = set([])

                # add xml
                xmls[subdir].add(os.path.join(subdir, file_name))

    # check if some jpgs are unlabeled
    unlabeled = check_labeling(jpgs, xmls)
    if len(unlabeled) > 0:
        for jpg in unlabeled:
            raise("An image is unlabelled. Please run: label {}.jpg".format(jpg))

    # Reset train and test folders
    if os.path.exists(TRAIN_IMAGES):
        shutil.rmtree(TRAIN_IMAGES)
    
    os.mkdir(TRAIN_IMAGES)

    if os.path.exists(TEST_IMAGES):
        shutil.rmtree(TEST_IMAGES)
    
    os.mkdir(TEST_IMAGES)

    # convert sets to list for indexing
    for label in jpgs:
        jpgs[label] = list(jpgs[label])
    
    for label in xmls:
        xmls[label] = list(xmls[label])

    # copy to training dir
    counter = 0
    while counter < trainees:
        for label in jpgs:
            # copy from path and remove from list
            shutil.copy(jpgs[label].pop() + '.jpg', TRAIN_IMAGES)

            # same with xmls
            shutil.copy(xmls[label].pop() + '.xml', TRAIN_IMAGES)
        
        counter+=1
    
    # copy remaining to testing dir
    for label in jpgs:
        while len(jpgs[label]) > 0:
            shutil.copy(jpgs[label].pop() + '.jpg', TEST_IMAGES)
            shutil.copy(xmls[label].pop() + '.xml', TEST_IMAGES)

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

    if os.name == 'nt':
        os.system('pip install pytz')

    print('Creating TF record for training...')
    os.system('python {} -x {} -l {} -o {}'.format(TF_RECORD_SCRIPT, TRAIN_IMAGES, LABEL_MAP, TRAIN_TF_RECORD))
    
    print('Creating TF record for testing...')
    os.system('python {} -x {} -l {} -o {}'.format(TF_RECORD_SCRIPT, TEST_IMAGES, LABEL_MAP, TEST_TF_RECORD))


'''------------------------------------------------------------------------------
update_pipeline_config

Description: Update Config for Transfer Learning
Arguments:
- number_of_labels: how many labels
- custom_model: the model name to be trained
- pretrained_model: pretrained model name
------------------------------------------------------------------------------'''
def update_pipeline_config(number_of_labels, custom_model, pretrained_model):
    print('Updating pipeline config...')

    # Copy Model Config to Training Folder
    pipeline_config_path = os.path.join(PRETRAINED_MODEL_PATH, pretrained_model, 'pipeline.config')
    custom_model_path = os.path.join(API_MODEL_PATH, custom_model)
    
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
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(PRETRAINED_MODEL_PATH, pretrained_model, 'checkpoint', 'ckpt-0')
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

'''------------------------------------------------------------------------------
train

Description: Train a model
Arguments:
- model: custom model name
- labels: annotations created in LabelImg
- folder: image folder
- sample: number of trainees from image folder
- pretrained: pretrained model name
- steps: number of steps in training
------------------------------------------------------------------------------'''
def train(model, labels, folder, sample, pretrained, steps):
    # do partition
    partition(folder, sample)

    # Create Label Map
    create_label_map(labels)

    # Create TF Records
    create_tf_records()

    # Update Config for Transfer Learning
    custom_model_path = update_pipeline_config(len(labels), model, pretrained)
    print(custom_model_path)

    if os.name == 'nt':
        os.system('pip install pycocotools-windows')
        os.system('pip install gin-config') 
        os.system('pip install tensorflow-addons')

    # Train
    os.system('python {} --model_dir={} --pipeline_config_path={} --num_train_steps={}'.format(TRAINING_SCRIPT, custom_model_path, os.path.join(custom_model_path, 'pipeline.config'), steps))
