from torchvision import models,transforms,datasets
import torch
from catfish_fit import CatfishModel
transforms = transforms.Compose([
 transforms.Resize([224,298]),
 transforms.ToTensor(),
 transforms.Normalize(mean=[0.485, 0.456, 0.406],
 std=[0.229, 0.224, 0.225] )
 ])
CatfishClasses = ["cat","fish"]
val_data_path = "/content/drive/MyDrive/Catfish_data/val"
val_data = datasets.ImageFolder(root=val_data_path,transform=transforms)
batch_size = 16
val_data_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size=batch_size)
