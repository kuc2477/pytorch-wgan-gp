from torchvision import datasets, transforms


_MNIST_TRAIN_TRANSFORMS = _MNIST_TEST_TRANSFORMS = [
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.Pad(2),
    transforms.ToTensor(),
]

_CIFAR_TRAIN_TRANSFORMS = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]

_CIFAR_TEST_TRANSFORMS = [
    transforms.ToTensor(),
]

_SHVN_TRAIN_TRANSFORMS = _SHVN_TEST_TRANSFORMS = [
    transforms.Scale(32),
    transforms.ToTensor(),
]


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
    )
}


DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'cifar100': {'size': 32, 'channels': 3, 'classes': 100},
    'shvn': {'size': 32, 'channels': 3},
}
