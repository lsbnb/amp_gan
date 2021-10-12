from torch import nn
from torch import optim


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class TransposedCBR(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride, padding):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.model(x)


class CRBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride, padding, bias=True):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel, stride, padding, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, in_dim, hidden_num, out_dim):
        super().__init__()
        self.model = nn.Sequential(
            TransposedCBR(in_dim, hidden_num * 8, (5, 1), (1, 1), (0, 0)),
            TransposedCBR(hidden_num * 8, hidden_num * 4, (4, 1), (2, 1), (1, 0)),
            TransposedCBR(hidden_num * 4, hidden_num * 2, (8, 1), (1, 1), (1, 0)),
            TransposedCBR(hidden_num * 2, hidden_num, (4, 1), (2, 1), (1, 0)),
            nn.ConvTranspose2d(hidden_num, 1, (1, out_dim), (1, 1), (0, 0), bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden_num):
        super().__init__()
        self.model = nn.Sequential(
            CRBlock(1, hidden_num, (1, in_dim), (1, 1), (0, 0), bias=False),
            CRBlock(hidden_num, hidden_num * 2, (4, 1), (2, 1), (1, 0), bias=False),
            CRBlock(hidden_num * 2, hidden_num * 4, (8, 1), (1, 1), (1, 0), bias=False),
            CRBlock(hidden_num * 4, hidden_num * 8, (4, 1), (2, 1), (1, 0), bias=False),
            nn.Conv2d(hidden_num * 8, 1, (5, 1), (1, 1), (0, 0), bias=False),
        )

    def forward(self, x):
        return self.model(x)


def get_model_and_optimizer(latent_size, encoded_num, hidden_size):
    generator = Generator(latent_size, hidden_size, encoded_num)
    generator.apply(weights_init)
    discriminator = Discriminator(encoded_num, hidden_size)
    discriminator.apply(weights_init)
    discrim_optim = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0, 0.9))
    gen_optim = optim.Adam(generator.parameters(), lr=0.0001, betas=(0, 0.9))
    return generator, discriminator, gen_optim, discrim_optim
