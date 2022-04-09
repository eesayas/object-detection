import os
from constants import LABELIMG_PATH, IMAGES_FOLDER
from git import Repo, RemoteProgress

# for showing git clone progress
class CloneProgress(RemoteProgress):
    def update(self, op_code, cur_count, max_count=None, message=''):
        if message:
            print('Cloning LabelImg. ' + message)

def label(folder):

    folder = IMAGES_FOLDER if folder == False else folder

    # mac user
    if os.name == 'posix':
        os.system('labelImg ' + folder)

    # windows user
    # TODO need testing
    if os.name =='nt':
        if not os.path.exists(LABELIMG_PATH):
            os.mkdir(LABELIMG_PATH)
            print('Cloning LabelImg. Please wait...')
            Repo.clone_from('https://github.com/tzutalin/labelImg', LABELIMG_PATH, progress=CloneProgress())
            os.system('cd ' + LABELIMG_PATH + ' && pyrcc5 -o libs/resources.py resources.qrc')

        os.system('cd ' + LABELIMG_PATH + ' && python labelImg.py ' + folder)
