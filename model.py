from torch import nn


class Critic(nn.Module):
    # TODO: NOT IMPLEMENTED YET
    pass


class Generator(nn.Module):
    # TODO: NOT IMPLEMENTED YET
    pass


class WGAN(nn.Module):
    def __init__(self, image_size):
        # TODO: NOT IMPLEMENTED YET
        self.image_size = image_size
        self.critic = Critic()
        self.generator = Generator()

    def forward(self, z, x=None):
        # generate the fake image from the given noise z.
        g = self.generator(z)

        # when training the critic. (-wasserstein distance between x and g)
        if x is not None:
            return -(self.critic(x)-self.critic(g))
        # when training the generator. (partial wasserstein distance)
        else:
            return -self.critic(g)

    def sample_image(self, size):
        # TODO: NOT IMPLEMENTED YET
        pass

    def sample_noise(self, size):
        # TODO: NOT IMPLEMENTED YET
        pass

    def gradient_penalty(self, x, g):
        # TODO: NOT IMPLEMENTED YET
        pass

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda
