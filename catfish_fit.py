import torchvision
from torchvision import models,transforms,datasets
import torch
import os
from PIL import Image
from urllib.request import urlopen
import numpy as np
import torch.optim as optim
transforms = transforms.Compose([
 transforms.Resize([224,298]),
 transforms.ToTensor(),
 transforms.Normalize(mean=[0.485, 0.456, 0.406],
 std=[0.229, 0.224, 0.225] )
 ])
CatfishModel  = models.resnet101(pretrained=True)
CatfishModel.fc = torch.nn.Sequential(torch.nn.Linear(transfer_model.fc.in_features,500),
 torch.nn.ReLU(),
 torch.nn.Linear(500,2))
for name, param in CatfishModel.named_parameters():
  if("bn" not in name and "fc" not in name):
    param.requires_grad = False
CatfishClasses = ["cat","fish"]
train_data_path = "C:/Users/stepan/PycharmProjects/Flaskserver/train"
train_data = datasets.ImageFolder(root=train_data_path,transform=transforms)
batch_size = 16
train_data_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle = True)
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
    print(target)
    optimizer.zero_grad()
    inputs = inputs.to(device)
    target = target.to(device)
    output = CatfishModel(inputs)
    print(output)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
torch.save(CatfishModel.state_dict(), 'C:/Users/stepan/PycharmProjects/Flaskserver/weights.pt')
