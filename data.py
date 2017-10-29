import torch
from torchvision import datasets, transforms


# ==================
# Dataset Transforms
# ==================

_MNIST_TRAIN_TRANSFORMS = _MNIST_TEST_TRANSFORMS = [
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.Pad(2),
    transforms.ToTensor(),
]

_CIFAR_TRAIN_TRANSFORMS = [
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]

_CIFAR_TEST_TRANSFORMS = [
    transforms.ToTensor(),
]

_SHVN_IMAGE_SIZE = 32
_SHVN_TRAIN_TRANSFORMS = _SHVN_TEST_TRANSFORMS = [
    transforms.Scale(_SHVN_IMAGE_SIZE),
    transforms.CenterCrop(_SHVN_IMAGE_SIZE),
    transforms.ToTensor(),
]

_LSUN_IMAGE_SIZE = 64
_LSUN_TRAIN_TRANSFORMS = _LSUN_TEST_TRANSFORMS = [
    transforms.Scale(_LSUN_IMAGE_SIZE),
    transforms.CenterCrop(_LSUN_IMAGE_SIZE),
    transforms.ToTensor(),
]


def _LSUN_COLLATE_FN(batch):
    return torch.stack([x[0] for x in batch])


# ========
# Datasets
# ========

TRAIN_DATASETS = {
    'mnist': lambda: datasets.MNIST(
        './datasets/mnist', train=True, download=True,
        transform=transforms.Compose(_MNIST_TRAIN_TRANSFORMS)
    ),
    'cifar10': lambda: datasets.CIFAR10(
        './datasets/cifar10', train=True, download=True,
        transform=transforms.Compose(_CIFAR_TRAIN_TRANSFORMS)
    ),
    'cifar100': lambda: datasets.CIFAR100(
        './datasets/cifar100', train=True, download=True,
        transform=transforms.Compose(_CIFAR_TRAIN_TRANSFORMS)
    ),
    'shvn': lambda: datasets.SVHN(
        './datasets/shvn', download=True, split='train',
        transform=transforms.Compose(_SHVN_TEST_TRANSFORMS)
    ),
    'lsun': lambda: datasets.LSUNClass(
        './datasets/lsun/bedroom_train',
        transform=transforms.Compose(_LSUN_TRAIN_TRANSFORMS)
    )
}

TEST_DATASETS = {
    'mnist': lambda: datasets.MNIST(
        './datasets/mnist', train=False,
        transform=transforms.Compose(_MNIST_TEST_TRANSFORMS)
    ),
    'cifar10': lambda: datasets.CIFAR10(
        './datasets/cifar10', train=False,
        transform=transforms.Compose(_CIFAR_TEST_TRANSFORMS)
    ),
    'cifar100': lambda: datasets.CIFAR100(
        './datasets/cifar100', train=False,
        transform=transforms.Compose(_CIFAR_TEST_TRANSFORMS)
    ),
    'shvn': lambda: datasets.SVHN(
        './datasets/shvn', download=True, split='test',
        transform=transforms.Compose(_SHVN_TEST_TRANSFORMS)
    ),
    'lsun': lambda: datasets.LSUN(
        './datasets/lsun/bedroom_test',
        transform=transforms.Compose(_LSUN_TRAIN_TRANSFORMS)
    )
}

DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'cifar100': {'size': 32, 'channels': 3, 'classes': 100},
    'shvn': {'size': _SHVN_IMAGE_SIZE, 'channels': 3},
    'lsun': {'size': _LSUN_IMAGE_SIZE, 'channels': 3,
             'collate_fn': _LSUN_COLLATE_FN},
}
