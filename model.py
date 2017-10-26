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
            kernel_size=3, stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(
            channel_size, channel_size*2,
            kernel_size=3, stride=2, padding=1
        )
        self.conv3 = nn.Conv2d(
            channel_size*2, channel_size*4,
            kernel_size=3, stride=2, padding=1
        )
        self.fc = nn.Linear((image_size//8)**2 * channel_size*4, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.view(-1, (self.image_size//8)**2 * self.channel_size*4)
        return self.fc(x)


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
        self.bn0 = nn.BatchNorm2d(channel_size*8)
        self.bn1 = nn.BatchNorm2d(channel_size*4)
        self.deconv1 = nn.ConvTranspose2d(
            channel_size*8, channel_size*4,
            kernel_size=4, stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm2d(channel_size*2)
        self.deconv2 = nn.ConvTranspose2d(
            channel_size*4, channel_size*2,
            kernel_size=4, stride=2, padding=1,
        )
        self.bn3 = nn.BatchNorm2d(channel_size)
        self.deconv3 = nn.ConvTranspose2d(
            channel_size*2, channel_size,
            kernel_size=4, stride=2, padding=1
        )
        self.deconv4 = nn.ConvTranspose2d(
            channel_size, image_channel_size,
            kernel_size=3, stride=1, padding=1,
        )

    def forward(self, z):
        g = F.relu(self.bn0(self.fc(z).view(
            z.size(0),
            self.channel_size*8,
            self.image_size//8,
            self.image_size//8,
        )))
        g = F.relu(self.bn1(self.deconv1(g)))
        g = F.relu(self.bn2(self.deconv2(g)))
        g = F.relu(self.bn3(self.deconv3(g)))
        g = self.deconv4(g)
        return F.tanh(g) if self.image_channel_size > 1 else F.sigmoid(g)


class WGAN(nn.Module):
    def __init__(self, label, z_size,
                 image_size, image_channel_size,
                 c_channel_size, g_channel_size):
        # configurations
        super().__init__()
        self.label = label
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

    @property
    def name(self):
        return (
            'WGAN-GP'
            '-z{z_size}'
            '-c{c_channel_size}'
            '-g{g_channel_size}'
            '-{label}-{image_size}x{image_size}x{image_channel_size}'
        ).format(
            z_size=self.z_size,
            c_channel_size=self.c_channel_size,
            g_channel_size=self.g_channel_size,
            label=self.label,
            image_size=self.image_size,
            image_channel_size=self.image_channel_size,
        )

    def c_loss(self, x, z, return_g=False):
        g = self.generator(z)
        c_x = self.critic(x).mean()
        c_g = self.critic(g).mean()
        l = -(c_x-c_g)
        return (l, g) if return_g else l

    def g_loss(self, z, return_g=False):
        g = self.generator(z)
        l = -self.critic(g).mean()
        return (l, g) if return_g else l

    def sample_image(self, size):
        return self.generator(self.sample_noise(size))

    def sample_noise(self, size):
        z = Variable(torch.randn(size, self.z_size))
        return z.cuda() if self._is_on_cuda() else z

    def gradient_penalty(self, x, g, lamda):
        assert x.size() == g.size()
        a = torch.rand(x.size(0), 1)
        a = a.cuda() if self._is_on_cuda() else a
        a = a\
            .expand(x.size(0), x.nelement()//x.size(0))\
            .contiguous()\
            .view(
                x.size(0),
                self.image_channel_size,
                self.image_size,
                self.image_size
            )
        interpolated = Variable(a*x.data + (1-a)*g.data, requires_grad=True)
        c = self.critic(interpolated)
        gradients = autograd.grad(
            c, interpolated, grad_outputs=(
                torch.ones(c.size()).cuda() if self._is_on_cuda() else
                torch.ones(c.size())
            ),
            create_graph=True,
            retain_graph=True,
        )[0]
        return lamda * ((1-gradients.norm(2, dim=1))**2).mean()

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda
