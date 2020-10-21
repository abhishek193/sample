import fastai
print(fastai.__version__)

from fastai.vision.all import *

classes = ['badminton', 'tennis']

'''
def getdata():
    for class_name in classes:
        folder = class_name
        file = 'download_tennis'
        dest = path/folder
        dest.mkdir(parents=True, exist_ok=True)
        download_images(path/file, dest, max_pics=200)
'''
def label_func(f): 
  return f[0] == 't'

import os
import shutil
files = os.listdir(os.getcwd()+'/train50/train')
path = os.getcwd() + '/train50/train'
test_files = os.listdir(os.getcwd())
for x in test_files:
    if x.endswith('.jpg'):
        files.append(x)
        shutil.copyfile(x, '/train50/train')
print(len(files))
os.chdir(path)

def train(files):
    dls = ImageDataLoaders.from_path_func(path, files, label_func, item_tfms=Resize(100))
    learn = cnn_learner(dls, resnet34, metrics=error_rate)
    learn.fine_tune(1)
    return learn

#getdata()
my_model = train(files)

#my_model.save('my_model')
