from flask import Flask, jsonify, request
from model import CatfishModel
from torchvision import transforms
import torch
import os
import requests
from PIL import Image
from urllib.request import urlopen
import numpy as np
import torchvision
transforms = transforms.Compose([
  transforms.Resize([224,298]),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406],
  std=[0.229, 0.224, 0.225])
 ])
CatfishClasses = ["cat","fish"]
def load_model():
  m = CatfishModel
  location = os.environ["CATFISH_MODEL_LOCATION"]
  m.load_state_dict(torch.load(location,map_location=torch.device('cpu')))
  return m
app = Flask(__name_
@app.route("/")
def status():
  return jsonify({"status": "ok"})
@app.route("/predict", methods=['GET', 'POST'])
def predict():
  img_url = request.args['image_url']
  img = Image.open(urlopen(img_url))
  img_tensor = transforms(img)[np.newaxis,:]
  model = load_model()
  prediction = model(img_tensor)
  print(prediction)
  predicted_class = CatfishClasses[torch.argmax(prediction)]
  return jsonify({"image": img_url, "prediction": predicted_class})
if __name__ == '__main__':
  os.environ['CATFISH_HOST'] = '127.0.0.1'
  os.environ['CATFISH_PORT'] = '8080'
  os.environ["CATFISH_MODEL_LOCATION"] = 'C:/Users/stepan/PycharmProjects/Flaskserver/weights.pt'
  app.run(host=os.environ["CATFISH_HOST"], port=os.environ["CATFISH_PORT"])
