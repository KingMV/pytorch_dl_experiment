from image_classification.vgg import vgg_model
from torchsummary import summary
from torch import optim
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torchvision
import os
from image_classification.vgg.utils import save_checkpoint

learning_rate = 1e-4

transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.199, 0.2010)),
                                      ])

transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.199, 0.2010)),
                                     ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


def train(network, dataLoader, epoches, save_dir, display_interval):
    network.cuda()
    network.train()

    # define the optimizer
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
    # define loss function
    criterion = nn.CrossEntropyLoss()
    best_acc = 0

    for ep in range(epoches):
        train_loss = 0
        correct = 0
        total_num = 0
        for batch_idx, (image, label) in enumerate(dataLoader):
            image = image.cuda()
            label = label.cuda()

            optimizer.zero_grad()

            output = network(image)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total_num += label.size(0)
            correct += predicted.eq(label).sum().item()
            if batch_idx % display_interval == 0:
                print('epoches:%04d  batch_idx:%04d  loss:%.2f  Acc:%.2f' % (
                    ep, batch_idx, train_loss / (batch_idx + 1), 100. * correct / total_num))
        acc = test(network, testloader)
        if acc > best_acc:
            best_acc = acc
            print('saving model checkpoints')
            state = {
                'net': network.state_dict(),
                'acc': best_acc,
                'epoch': ep
            }
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            save_checkpoint(state, os.path.join(save_dir, 'e{0}_checkpoint_acc{1}.pth'.format(ep, best_acc)))


def test(network, dataLoader):
    network.cuda()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(dataLoader):
            image = image.cuda()
            label = label.type(torch.LongTensor).cuda()

            outputs = network(image)
            loss = criterion(outputs, label)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            print('test result: loss:%.2f  Acc:%.2f' % (test_loss / total, 100. * correct / total))
    return 100. * correct / total


if __name__ == '__main__':
    num_class = 10
    vgg_model_path = './image_classification/model/vgg/vgg16-397923af.pth'
    model_save_dir = './model'
    net = vgg_model.vgg16(pretrained=True, model_path=vgg_model_path).cuda()
    # adjust the architecture of network
    x = torch.rand(2, 3, 32, 32).cuda()
    y = net.features(x)
    flat_feature_dim = y.size(1) * y.size(2) * y.size(3)
    net.classifier = nn.Sequential(nn.Linear(flat_feature_dim, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, num_class)
                                   )

    train(net, trainloader, 100, model_save_dir, 50)
    print('train done!')

    # net.classifier = nn.Linear()
