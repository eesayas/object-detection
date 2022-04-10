import os
import shutil

'''--------------------------------------------------------
checkLabeling
Description: check if all jpgs are labeled
Args:
- jpgs: dictonary of sets ( ex: { label: { jpg1, jpg2 } } )
- xmls: dictonary of sets ( ex: { label: { xml1, xml2 } } )
--------------------------------------------------------'''
def checkLabeling(jpgs, xmls):
    
    unlabeled = []

    for label in jpgs:
        for jpg in jpgs[label].difference(xmls[label]):
            unlabeled.append(jpg)
    
    return unlabeled
        
        
def partition(flags):
    jpgs = {}
    xmls = {}
    valid = True
    train = flags['train']
    index = 0

    for subdir, dirs, files in os.walk(flags['folder']):
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
    unlabeled = checkLabeling(jpgs, xmls)
    if len(unlabeled) > 0:
        for jpg in unlabeled:
            print("An image is unlabelled. Please run: label {}.jpg".format(jpg))

    # create train and test dir
    if not os.path.exists('train'):
        os.mkdir('train')

    if not os.path.exists('test'):
        os.mkdir('test')

    # convert sets to list for indexing
    for label in jpgs:
        jpgs[label] = list(jpgs[label])
    
    for label in xmls:
        xmls[label] = list(xmls[label])

    # move to training dir
    while index < train:

        for label in jpgs:
            shutil.move(jpgs[label][index] + '.jpg', 'train')
            shutil.move(xmls[label][index] + '.xml', 'train')

        index+=1
    
    # move remaining to testing dir
    for subdir, dirs, files in os.walk(flags['folder']):
        for file in files:
            file_name, file_extension = os.path.splitext(file)
            if file_extension in ['.jpg', '.xml']:
                shutil.move(os.path.join(subdir, file), 'test')  