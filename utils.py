import os
import os.path
import torch
from torch.utils.data import DataLoader
from torch.nn import init
import torchvision


def get_data_loader(dataset, batch_size, cuda=False):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        **({'num_workers': 1, 'pin_memory': True} if cuda else {})
    )


def save_checkpoint(model, model_dir, iteration):
    path = os.path.join(model_dir, model.name)

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save({
        'state': model.state_dict(),
        'iteration': iteration,
    }, path)

    # notify that we successfully saved the checkpoint.
    print('=> saved the model {name} to {path}'.format(
        name=model.name, path=path
    ))

    return iteration


def load_checkpoint(model, model_dir):
    path = os.path.join(model_dir, model.name)

    # load the checkpoint.
    checkpoint = torch.load(path)
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=path
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    iteration = checkpoint['iteration']
    return iteration


def test_model(model, sample_size, path):
    os.makedirs(os.path.basename(path), exists_ok=True)
    torchvision.utils.save_image(
        model.sample_image(sample_size),
        path
    )
    print('=> generated sample images at "{}".'.format(path))


def xavier_initialize(model):
    modules = [
        m for n, m in model.named_modules() if
        'conv' in n or 'fc' in n
    ]

    parameters = [
        p for
        m in modules for
        p in m.parameters() if
        p.dim() >= 2
    ]

    for p in parameters:
        init.xavier_normal(p)


def gaussian_intiailize(model, std=.01):
    for p in model.parameters():
        init.normal(p, std=std)
