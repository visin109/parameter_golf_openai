from torchvision import datasets
import numpy as np

def load_cifar10():
    train = datasets.CIFAR10(root="./data", train=True, download=True)
    test = datasets.CIFAR10(root="./data", train=False, download=True)

    X_train = np.array(train.data)   # (50000, 32, 32, 3)
    y_train = np.array(train.targets)

    X_test = np.array(test.data)     # (10000, 32, 32, 3)
    y_test = np.array(test.targets)

    return X_train, y_train, X_test, y_test