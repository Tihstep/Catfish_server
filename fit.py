import torchvision
from torchvision import models,transforms,datasets
import torch
import os
from PIL import Image
from urllib.request import urlopen
import numpy as np
import torch.optim as optim
img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Felis_silvestris_silvestris.jpg/208px-Felis_silvestris_silvestris.jpg"
img = Image.open(urlopen(img_url))
transforms = transforms.Compose([
 transforms.Resize([224,298]),
 transforms.ToTensor(),
 transforms.Normalize(mean=[0.485, 0.456, 0.406],
 std=[0.229, 0.224, 0.225] )
 ])
img_tensor = transforms(img)[np.newaxis,:]
transfer_model = models.resnet101(pretrained=True)
for name, param in transfer_model.named_parameters():
  if("bn" not in name):
    param.requires_grad = False
CatfishClasses = ["cat","fish"]
CatfishModel = models.resnet101()
CatfishModel.fc = torch.nn.Sequential(torch.nn.Linear(transfer_model.fc.in_features,500),
 torch.nn.ReLU(),
 torch.nn.Linear(500,2))
train_data_path = "/content/drive/MyDrive/Catfish_data/train"
train_data = datasets.ImageFolder(root=train_data_path,transform=transforms)
batch_size = 16
train_data_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(CatfishModel.parameters(), lr=0.001)
if torch.cuda.is_available():
 device = torch.device("cuda")
else:
 device = torch.device("cpu")
CatfishModel.to(device)
for _ in range(10):
  for batch in train_data_loader:
    inputs, target = batch
    print(inputs.shape,target)
    optimizer.zero_grad()
    inputs = inputs.to(device)
    target = target.to(device)
    output = CatfishModel(inputs)
    print(output)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
