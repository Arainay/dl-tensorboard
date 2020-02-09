from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from numpy import transpose


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def get_train_set_and_loader():
    train_set = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    train_loader = train_loader = DataLoader(
        train_set,
        batch_size=4,
        shuffle=True
    )

    return [train_set, train_loader]


def get_test_set_and_loader():
    test_set = torchvision.datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_set,
        batch_size=4,
        shuffle=True
    )

    return [test_set, test_loader]


def show_image(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap='Greys')
    else:
        plt.imshow(transpose(npimg, (1, 2, 0)))
