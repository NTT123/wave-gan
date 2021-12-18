import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from utils import get_padding, init_weights

LRELU_SLOPE = 0.1


class UpsampleNet(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        num_conv = 5
        dim = h.upsample_initial_channel
        self.input_conv = weight_norm(Conv1d(80, dim, 1, 1))
        self.input_conv.apply(init_weights)
        self.convs = nn.ModuleList()
        for i in range(num_conv):
            self.convs.append(
                weight_norm(Conv1d(dim, dim, 3, 1, padding="same", dilation=2 ** i))
            )
        self.convs.apply(init_weights)

        self.upsample_conv_1 = weight_norm(ConvTranspose1d(dim, dim, 4, 4, padding=0))
        self.upsample_conv_2 = weight_norm(ConvTranspose1d(dim, dim, 4, 4, padding=0))
        self.upsample_conv_1.apply(init_weights)
        self.upsample_conv_2.apply(init_weights)

    def forward(self, x):
        x = self.input_conv(x)
        for conv in self.convs:
            residual = x
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = (x + residual) * math.sqrt(0.5)
        x = self.upsample_conv_1(x)
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.upsample_conv_2(x)
        x = F.leaky_relu(x, LRELU_SLOPE)
        # x16
        N, D, L = x.shape
        x = torch.reshape(x, (N, D, L, 1))
        x = torch.tile(x, (1, 1, 1, 16))
        x = torch.reshape(x, (N, D, L * 16))
        return x


class WaveNetBlock(torch.nn.Module):
    def __init__(self, h, num_layer=10):
        super().__init__()
        self.h = h
        self.convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        dim = h.upsample_initial_channel
        for i in range(num_layer):
            self.convs.append(
                weight_norm(
                    nn.Conv1d(dim, 2 * dim, 3, 1, padding="same", dilation=2 ** i)
                )
            )
            self.skip_convs.append(weight_norm(nn.Conv1d(dim, dim, 1, 1)))
        self.convs.apply(init_weights)
        self.skip_convs.apply(init_weights)

    def forward(self, x):
        skips = []
        for conv, skip_conv in zip(self.convs, self.skip_convs):
            residual = x
            x = conv(x)
            x_tanh, x_sigmoid = torch.chunk(x, 2, dim=1)
            x = torch.tanh(x_tanh) * torch.sigmoid(x_sigmoid)
            x = skip_conv(x)
            skips.append(x)
            x = x + residual
        return x, skips


class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.upsample = UpsampleNet(h)
        num_blocks = 3
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(WaveNetBlock(h))

        dim = h.upsample_initial_channel
        self.output = nn.Sequential(
            weight_norm(nn.Conv1d(dim, dim, 1, 1)),
            torch.nn.ReLU(),
            weight_norm(nn.Conv1d(dim, dim, 1, 1)),
            torch.nn.ReLU(),
            weight_norm(nn.Conv1d(dim, 1, 1, 1)),
        )
        self.output.apply(init_weights)

    def forward(self, x):
        x = self.upsample(x)
        skips = None
        for block in self.blocks:
            x, skip = block(x)
            if skips is None:
                skips = sum(skip)
            else:
                skips = skips + sum(skip)
        x = skips
        x = torch.relu(x)
        x = self.output(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(2),
                DiscriminatorP(3),
                DiscriminatorP(5),
                DiscriminatorP(7),
                DiscriminatorP(11),
            ]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorS(use_spectral_norm=True),
                DiscriminatorS(),
                DiscriminatorS(),
            ]
        )
        self.meanpools = nn.ModuleList(
            [AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
