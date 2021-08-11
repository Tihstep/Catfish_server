from torchvision import models
from torch import nn
CatfishModel = models.resnet101()
CatfishModel.fc = nn.Sequential(nn.Linear(CatfishModel.fc.in_features,500),
 nn.ReLU(),
 nn.Linear(500,2))
for name, param in CatfishModel.named_parameters():
 if("bn" not in name and "fc" not in name):
  param.requires_grad = False
CatfishClasses = ["cat","fish"]
