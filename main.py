import torchvision.models as models
from torchsummary import summary

# call the model
vgg16 = models.vgg16().cuda()

summary(vgg16, (3, 224, 224))
# for m in vgg16.modules():
#     print(m)
# print(vgg16.modules())
