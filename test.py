import fastai
print(fastai.__version__)

from fastai.vision.all import *
import sys
import fast
import os

def predict(model, image):
    out = model.predict(image)
    return out

print(os.getcwd())
out = predict(fast.my_model, '1.jpg')
print(out)