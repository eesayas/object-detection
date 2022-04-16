import os

'''----------------------------------------------------------------
find_latest_ckpt

Description: This will return the latest ckpt index from training
Arguments:
- model_path: the path to the trained model

Return: int 
-----------------------------------------------------------------'''
def find_latest_ckpt(model_path):
    max_ckpt = 0

    for subdir, dirs, files in os.walk(model_path):
        for file in files:
            file_name, file_extension = os.path.splitext(file)
            if 'ckpt' in file_name:
                ckpt = int(file_name.split('-')[1])
                if ckpt > max_ckpt:
                    max_ckpt = ckpt

    return max_ckpt