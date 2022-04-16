import os

def label(folder):

    # mac user
    if os.name == 'posix':
        os.system('labelImg {}'.format(folder))

    # windows user
    if os.name == 'nt':
       print('need implementation for windows')