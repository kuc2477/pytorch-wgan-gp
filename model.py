import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import autograd


class Critic(nn.Module):
    def __init__(self, image_size, image_channel_size, channel_size):
        # configurations
        super().__init__()
        self.image_size = image_size
        self.image_channel_size = image_channel_size
        self.channel_size = channel_size

        # layers
        self.conv1 = nn.Conv2d(
            image_channel_size, channel_size,
            kernel_size=3, padding=1, stride=2
        )
        self.conv2 = nn.Conv2d(
            image_channel_size, channel_size*2,
            kernel_size=3, padding=1, stride=2
        )
        self.conv3 = nn.Conv2d(
            image_channel_size, channel_size*4,
            kernel_size=3, padding=1, stride=2
        )
        self.fc = nn.Linear((image_size//8)**2 * channel_size*4, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.fc(x.view(x.size(0), -1))


class Generator(nn.Module):
    def __init__(self, z_size, image_size, image_channel_size, channel_size):
        # configurations
        super().__init__()
        self.z_size = z_size
        self.image_size = image_size
        self.image_channel_size = image_channel_size
        self.channel_size = channel_size

        # layers
        self.fc = nn.Linear(z_size, (image_size//8)**2 * channel_size*8)
        self.bn1 = nn.BatchNorm2d(channel_size*4)
        self.deconv1 = nn.ConvTranspose2d(
            channel_size*8, channel_size*4,
            kernel_size=3, padding=1, stride=2,
        )
        self.bn2 = nn.BatchNorm2d(channel_size*2)
        self.deconv2 = nn.ConvTranspose2d(
            channel_size*4, channel_size*2,
            kernel_size=3, padding=1, stride=2,
        )
        self.bn3 = nn.BatchNorm2d(channel_size)
        self.deconv3 = nn.ConvTranspose2d(
            channel_size*2, channel_size,
            kernel_size=3, padding=1, stride=2,
        )
        self.deconv4 = nn.ConvTranspose2d(
            channel_size, image_channel_size,
            kernel_size=3, padding=1, stride=1,
        )

    def forward(self, z):
        g = self.fc(z).view(
            z.size(0),
            self.channel_size*8,
            self.image_size//8,
            self.image_size//8,
        )
        g = F.relu(self.bn1(self.deconv1(g)))
        g = F.relu(self.bn2(self.deconv2(g)))
        g = F.relu(self.bn3(self.deconv3(g)))
        return F.tanh(self.deconv4(g))


class WGAN(nn.Module):
    def __init__(self, label, z_size,
                 image_size, image_channel_size,
                 c_channel_size, g_channel_size):
        # configurations
        super().__init__()
        self.z_size = z_size
        self.image_size = image_size
        self.image_channel_size = image_channel_size
        self.c_channel_size = c_channel_size
        self.g_channel_size = g_channel_size

        # components
        self.critic = Critic(
            image_size=self.image_size,
            image_channel_size=self.image_channel_size,
            channel_size=self.c_channel_size,
        )
        self.generator = Generator(
            z_size=self.z_size,
            image_size=self.image_size,
            image_channel_size=self.image_channel_size,
            channel_size=self.g_channel_size,
        )

    def forward(self, z, x=None):
        # generate the fake image from the given noise z.
        g = self.generator(z)

        # when training the critic. (-wasserstein distance between x and g)
        if x is not None:
            return -(self.critic(x)-self.critic(g))
        # when training the generator. (wasserstein distance without x)
        else:
            return -self.critic(g)

    def sample_image(self, size):
        return self.generator(self.sample_noise(size))

    def sample_noise(self, size):
        z = Variable(torch.randn(size, self.z_size))
        return z.cuda() if self._is_on_cuda() else z

    def gradient_penalty(self, x, g, lamda):
        assert x.size() == g.size()
        a = Variable(torch.rand(x.size(0), 1, 1, 1))
        a = a.expand_as(x)
        a = a.cuda() if self._is_on_cuda() else a

        interpolated = a*x + (1-a)*g
        interpolated.requires_grad = True
        c = self.critic(interpolated)
        gradients = autograd.grad(c, interpolated)
        return lamda*((1-gradients.norm(2, dim=1))**2).mean()

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda
