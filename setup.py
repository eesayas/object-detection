import os
from gitclone import gitclone
from constants import API_MODEL_PATH, PROTOC_MAC_URL, PROTOC_MAC_ZIP, PROTOC_PATH, PROTOC_WIN_URL, PROTOC_WIN_ZIP
import shutil
import wget

'''------------------------------------------------------------------------------
api_model

Description: Download the api model
------------------------------------------------------------------------------'''
def api_model():
    if os.path.exists(API_MODEL_PATH):
        print('Tensorflow API Model already downloaded')
        return
    
    os.mkdir(API_MODEL_PATH)
    gitclone('https://github.com/tensorflow/models', API_MODEL_PATH)

'''------------------------------------------------------------------------------
protoc

Description: Download and run protoc
------------------------------------------------------------------------------'''
def protoc():
    if os.path.exists(PROTOC_PATH):
        shutil.rmtree(PROTOC_PATH)
    
    os.mkdir(PROTOC_PATH)

    if os.name == 'posix':
        # download protoc for mac
        os.system('wget {}'.format(PROTOC_MAC_URL))
        shutil.unpack_archive(PROTOC_MAC_ZIP, PROTOC_PATH)
        os.remove(PROTOC_MAC_ZIP)

        # permissions 
        os.system('chmod +x protoc/bin/protoc')

        # add protoc to path
        os.environ['PATH'] = os.path.join(os.getcwd(), 'protoc', 'bin') + ':' + os.environ['PATH']

        # run
        os.system('cd {}/models/research && protoc object_detection/protos/*.proto --python_out=.'.format(os.getcwd()))

    if os.name == 'nt':
        # download protoc for windows
        wget.download(PROTOC_WIN_URL)
        shutil.unpack_archive(PROTOC_WIN_ZIP, PROTOC_PATH)
        os.remove(PROTOC_WIN_ZIP)

        # add protoc to path
        os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(PROTOC_PATH, 'bin'))  

        # run
        os.system('cd {}/models/research && protoc object_detection/protos/*.proto --python_out=.'.format(os.getcwd()))

'''------------------------------------------------------------------------------
installTF

Description: Install tensorflow and its dependencies
------------------------------------------------------------------------------'''
def installTF():
    if not os.path.exists('models'):
        raise Exception('Models from Tensorflow Garden must be downloaded first')

    currentdir = os.getcwd()

    if os.name == 'posix':
        os.chdir(os.path.join(currentdir, 'models', 'research'))
        shutil.copy(os.path.join('object_detection', 'packages', 'tf2', 'setup.py'), '.')
        os.system('python -m pip install .')

    if os.name == 'nt':
        print('Windows implentation to be done')    

    os.chdir(currentdir) # reset

'''------------------------------------------------------------------------------
verify

Description: Verify the install of Tensorflow
------------------------------------------------------------------------------'''
def verify():
    verification_script = os.path.join(API_MODEL_PATH, 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
    os.system('python {}'.format(verification_script))


'''------------------------------------------------------------------------------
setup

Description: Aggregate all functions above
------------------------------------------------------------------------------'''
def setup():
    # Download TF Models
    api_model()
    
    # Install and run protoc
    protoc()
    # installTF()

    # Verify Tensorflow
    # verify()

setup()
