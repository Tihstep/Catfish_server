from torchvision import models
from torch import nn
transfer_model = models.resnet50(pretrained=True)
for name, param in transfer_model.named_parameters():
 if("bn" not in name):
  param.requires_grad = False
CatfishClasses = ["cat","fish"]
CatfishModel = models.resnet50()
CatfishModel.fc = nn.Sequential(nn.Linear(transfer_model.fc.in_features,500),
 nn.ReLU(),
 nn.Dropout(), nn.Linear(500,2))