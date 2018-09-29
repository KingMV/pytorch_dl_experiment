import torchvision
import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import numpy as np

transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.199, 0.2010)),
                                      ])

trainset = torchvision.datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

for img, label in trainloader:
    print(img.size())
    # print(label)
# plt.imshow(image)
# print(data.size(), label)
# plt.show()
