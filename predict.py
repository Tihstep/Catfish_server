img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Felis_silvestris_silvestris.jpg/208px-Felis_silvestris_silvestris.jpg"
img = Image.open(urlopen(img_url))
img_tensor = transforms(img)[np.newaxis,:]
img_tensor = img_tensor.to('cuda')
print(type(img_tensor))
prediction = CatfishModel(img_tensor)
predicted_class = CatfishClasses[torch.argmax(prediction)]
print(prediction, predicted_class)
