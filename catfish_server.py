from flask import Flask, jsonify, request
from model import CatfishModel
from torchvision import transforms
import torch
import os
import requests
from PIL import Image
from urllib.request import urlopen
def load_model():
 return model
app = Flask(__name__)
@app.route("/")
def status():
 return jsonify({"status": "ok"})
@app.route("/predict", methods=['GET', 'POST'])
def predict():
 img_url = request.args['image_url']
 img_tensor = Image.open(urlopen(img_url))
 prediction = CatfishModel(img_tensor)
 predicted_class = CatfishClasses[torch.argmax(prediction)]
 return jsonify({"image": img_url, "prediction": predicted_class})
if __name__ == '__main__':
 os.environ['CATFISH_HOST'] = '127.0.0.1'
 os.environ['CATFISH_PORT'] = '8080'
 app.run(host=os.environ["CATFISH_HOST"], port=os.environ["CATFISH_PORT"])