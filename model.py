import kfserving
from torchvision import models, transforms
from typing import List, Dict
import torch
from PIL import Image
import base64
import io
import fastai
from fastai.vision.all import *
import os
import shutil

class KFServingSampleModel(kfserving.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False

    def load(self):
        files = os.listdir(os.getcwd()+'/train50/train')
        path = os.getcwd() + '/train50/train'
        test_files = os.listdir(os.getcwd())
        for x in test_files:
            if x.endswith('.jpg'):
                files.append(x)
                shutil.copy(x, path)
        
        os.chdir(path)
        dls = ImageDataLoaders.from_path_func(path, files, label_func, item_tfms=Resize(100))
        learn = cnn_learner(dls, resnet34, metrics=error_rate)
        self.model = learn.fine_tune(1)

        self.ready = True

    def predict(self, request: Dict) -> Dict:
        inputs = request["instances"]

        # Input follows the Tensorflow V1 HTTP API for binary values
        # https://www.tensorflow.org/tfx/serving/api_rest#encoding_binary_values
        data = inputs[0]["image"]["b64"]

        raw_img_data = base64.b64decode(data)
        input_image = Image.open(io.BytesIO(raw_img_data))

        out = self.model.predict(image)

        return {"predictions": out}


if __name__ == "__main__":
    model = KFServingSampleModel("kfserving-model")
    model.load()
    kfserving.KFServer(workers=1).start([model])