from PIL import Image
import io
import os
import json
import requests

import torch
import pytorch_lightning as pl
import torch.nn as nn
import clip
from PIL import Image
import aiohttp


#####  This script will predict the aesthetic score for this image file:


def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


class Predictor:
    def __init__(self, model_path='./sac+logos+ava1-l14-linearMSE.pth'):
        self.model = MLP(768)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=self.device) # RN50x64  

    def predict_img_url(self, url, normalize=True):
        pil_image = Image.open(io.BytesIO(requests.get(url).content))
        return self.predict_img(pil_image, normalize=normalize)
    
    async def predict_img_url_async(self, url, normalize=True):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    image_data = await response.read()
                    pil_image = Image.open(io.BytesIO(image_data))
                    return self.predict_img(pil_image, normalize=normalize)
                else:
                    # Handle error cases here, e.g., return an error code or raise an exception.
                    return None
    
    def predict_img_local_path(self, img_path, normalize=True):
        pil_image = Image.open(img_path)
        return self.predict_img(pil_image, normalize=normalize)

    def predict_img(self, pil_img, normalize=True):
        image = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
        im_emb_arr = normalized(image_features.cpu().detach().numpy())
        prediction = self.model(torch.from_numpy(im_emb_arr).to(self.device).type(torch.cuda.FloatTensor))
        if normalize:
            prediction = torch.sigmoid(prediction)
        return prediction[0].item()
    

if __name__ == '__main__':
    image_path = './image_1.jpeg'
    predictor = Predictor()
    # print(predictor.predict_img_local_path(image_path))
    # image_path = './image_2.jpeg'
    # print(predictor.predict_img_local_path(image_path))
    # image_path = './image_3.jpeg'
    # print(predictor.predict_img_local_path(image_path))
    images = [
        'https://lh5.googleusercontent.com/p/AF1QipMv3b_BhZhkE2U1qMzN8tRv6x0WagBNoOVtyhU=s0',
        'https://lh5.googleusercontent.com/p/AF1QipPK4vnsiL81l8EN-9lMJ1Zc62l82UMJGh2vzR8n=s0',
        'https://lh5.googleusercontent.com/p/AF1QipOT9tywcIYGcVS1w3eVurVGregRFo5RfZKpHyWq=s0',
        'https://lh5.googleusercontent.com/p/AF1QipOAJdpbiLIlaQuzmQvfO4MJUE7tU5QZmCBWe5cA=s0',
        'https://lh5.googleusercontent.com/p/AF1QipN1YXw0-IBRc6gkDH4DLsO2eHwHZVEVmHkPHOS0=s0'
    ]
    for img in images:
        print(predictor.predict_img_url(img))

