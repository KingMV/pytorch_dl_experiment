import torchvision.models as models
from torchsummary import summary
# please install torchsummary package by cmd: pip install torchsummary
# call the model
vgg16 = models.vgg16().cuda()
# print the information of model it is similar with keras.
summary(vgg16, (3, 224, 224))