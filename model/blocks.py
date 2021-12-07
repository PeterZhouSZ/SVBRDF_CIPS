import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

import random

# self-defined crop function
def mycrop(x, size, center=False, rand0=None, tileable=True):

    b,c,h,w = x.shape
    if center:
        if rand0 is None:
            w0 = (w - size)*0.5
            h0 = (h - size)*0.5
            w0 = int(w0)
            h0 = int(h0)
        else:
            h0 = rand0[0]
            w0 = rand0[1]

        print('center: ', w0, h0)
        if w0+size>w or h0+size>h:
            raise ValueError('value error of w0')

        return x[:,:,w0:w0+self.size,h0:h0+self.size]

    if not tileable:
        if rand0 is None:
            w0 = random.randint(0,w-size)
            h0 = random.randint(0,h-size)
        else:
            h0 = rand0[0]
            w0 = rand0[1]

        if w0+size>w or h0+size>h:
            raise ValueError('value error of w0')
        print('rand: ', w0, h0)

        return x[:,:,w0:w0+size,h0:h0+size]

    else:
        if rand0 is None:
            w0 = random.randint(-size,w)
            h0 = random.randint(-size,h)
        else:
            h0 = rand0[0]
            w0 = rand0[1]

        # print('rand tile: ', h0, w0)

        wc = w0 + size
        hc = h0 + size

        p = torch.zeros((b,c,size,size), device='cuda')

        # seperate crop and stitch them manually
        # [7 | 8 | 9]
        # [4 | 5 | 6]
        # [1 | 2 | 3]
        # 1
        if h0<=0 and w0<=0:
            p[:,:,0:-h0,0:-w0] = x[:,:, h+h0:h, w+w0:w]
            p[:,:,-h0:,0:-w0] = x[:,:, 0:hc, w+w0:w]
            p[:,:,0:-h0,-w0:] = x[:,:, h+h0:h, 0:wc]
            p[:,:,-h0:,-w0:] = x[:,:, 0:hc, 0:wc]
        # 2
        elif h0<=0 and (w0<w-size and w0>0):
            p[:,:,0:-h0,:] = x[:,:, h+h0:h,w0:wc]
            p[:,:,-h0:,:] = x[:,:, 0:hc, w0:wc]
        # 3
        elif h0<=0 and w0 >=w-size:
            p[:,:,0:-h0,0:w-w0] = x[:,:, h+h0:h, w0:w]
            p[:,:,-h0:,0:w-w0] = x[:,:, 0:hc, w0:w]
            p[:,:,0:-h0,w-w0:] = x[:,:, h+h0:h, 0:wc-w]
            p[:,:,-h0:,w-w0:] = x[:,:, 0:hc, 0:wc-w]

        # 4
        elif (h0>0 and h0<h-size) and w0<=0:
            p[:,:,:,0:-w0] = x[:,:, h0:hc, w+w0:w]
            p[:,:,:,-w0:] = x[:,:, h0:hc, 0:wc]
        # 5
        elif (h0>0 and h0<h-size) and (w0<w-size and w0>0):
            p = x[:,:, h0:hc, w0:wc]
        # 6
        elif (h0>0 and h0<h-size) and w0 >=w-size:
            p[:,:,:,0:w-w0] = x[:,:, h0:hc, w0:w]
            p[:,:,:,w-w0:] = x[:,:, h0:hc, 0:wc-w]

        # 7
        elif h0 >=h-size and w0<=0:
            p[:,:,0:h-h0,0:-w0] = x[:,:, h0:h, w+w0:w]
            p[:,:,h-h0:,0:-w0] = x[:,:, 0:hc-h, w+w0:w]
            p[:,:,0:h-h0,-w0:] = x[:,:, h0:h, 0:wc]
            p[:,:,h-h0:,-w0:] = x[:,:, 0:hc-h, 0:wc]
        # 8
        elif h0 >=h-size and (w0<w-size and w0>0):
            p[:,:,0:h-h0,:] = x[:,:, h0:h,w0:wc]
            p[:,:,h-h0:,:] = x[:,:, 0:hc-h, w0:wc]
        # 9
        elif h0 >=h-size and w0 >=w-size:
            p[:,:,0:h-h0,0:w-w0] = x[:,:, h0:h, w0:w]
            p[:,:,h-h0:,0:w-w0] = x[:,:, 0:hc-h, w0:w]
            p[:,:,0:h-h0,w-w0:] = x[:,:, h0:h, 0:wc-w]
            p[:,:,h-h0:,w-w0:] = x[:,:, 0:hc-h, 0:wc-w]

        del x

        return p



class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class PatembNet(nn.Module):
    def __init__(self, emb_pat, in_pat_c, n_layers=3):
        super().__init__()

        # hard code the hidden dim
        hidden_dim = in_pat_c
        self.n_layers = n_layers
        if emb_pat=='net_conv':
            self.net = nn.ModuleList()
            for i in range(self.n_layers):
                in_channels = 1 if i==0 else hidden_dim
                self.net.append(nn.Conv2d(in_channels, hidden_dim, 5, padding=2, padding_mode='circular'))

        # elif emb_pat=='net_Unet':
        #     self.net = nn.ModuleList()
        #     for i in range(self.n_layers):
        #         in_channels = 1 if i==0 else hidden_dim
        #         self.net.append(nn.Conv2d(in_channels, hidden_dim, 5, padding=2))

        # elif emb_pat=='net_enco':


        self.leakyrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):

        for i in range(self.n_layers):
            if i!=self.n_layers-1:
                x = self.leakyrelu(self.net[i](x))
            else:
                x = self.tanh(self.net[i](x))

        return x



class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)
        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)
        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)
        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)
        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style, style_plus = False):
        batch, in_channel, height, width = input.shape
        # print('style_plus ', style_plus)
        if not style_plus:
            style = self.modulation(style).view(batch, 1, in_channel, 1, 1)

        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out, style


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
        activation=None,
        downsample=False,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
            downsample=downsample,
        )

        self.activation = activation
        self.noise = NoiseInjection()
        if activation == 'sinrelu':
            self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
            self.activate = ScaledLeakyReLUSin()
        elif activation == 'sin':
            self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
            self.activate = SinActivation()
        else:
            self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None, style_plus=False):
        # print('styled Conv: ', style_plus)
        out, style = self.conv(input, style, style_plus=style_plus)
        out = self.noise(out, noise=noise)
        if self.activation == 'sinrelu' or self.activation == 'sin':
            out = out + self.bias
        out = self.activate(out)

        return out, style


class ToRGB(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.upsample = upsample
        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, out_channel, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input, style, skip=None, style_plus=False):
        out, style = self.conv(input, style, style_plus=style_plus)
        out = out + self.bias
        # print('rgb styled Conv: ', style_plus)

        if skip is not None:
            if self.upsample:
                skip = self.upsample(skip)

            out = out + skip

        return out, style


class EqualConvTranspose2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv_transpose2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[0]}, {self.weight.shape[1]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        upsample=False,
        padding="zero",
    ):
        layers = []

        self.padding = 0
        stride = 1

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2

        if upsample:
            layers.append(
                EqualConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=0,
                    stride=2,
                    bias=bias and not activate,
                )
            )

            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

        else:
            if not downsample:
                if padding == "zero":
                    self.padding = (kernel_size - 1) // 2

                elif padding == "reflect":
                    padding = (kernel_size - 1) // 2

                    if padding > 0:
                        layers.append(nn.ReflectionPad2d(padding))

                    self.padding = 0

                elif padding != "valid":
                    raise ValueError('Padding should be "zero", "reflect", or "valid"')

            layers.append(
                EqualConv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                )
            )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], kernel_size=3, downsample=True):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, kernel_size)
        self.conv2 = ConvLayer(in_channel, out_channel, kernel_size, downsample=downsample)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=downsample, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class ConLinear(nn.Module):
    def __init__(self, ch_in, ch_out, is_first=False, bias=True):
        super(ConLinear, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, bias=bias)
        if is_first:
            nn.init.uniform_(self.conv.weight, -np.sqrt(9 / ch_in), np.sqrt(9 / ch_in))
        else:
            nn.init.uniform_(self.conv.weight, -np.sqrt(3 / ch_in), np.sqrt(3 / ch_in))

    def forward(self, x):
        return self.conv(x)


class SinActivation(nn.Module):
    def __init__(self,):
        super(SinActivation, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class LFF(nn.Module):
    def __init__(self, hidden_size, ):
        super(LFF, self).__init__()
        self.ffm = ConLinear(2, hidden_size, is_first=True)
        self.activation = SinActivation()

    def forward(self, x):
        x = self.ffm(x)
        x = self.activation(x)
        return x

class Myembed(nn.Module):
    def __init__(self, N = 512):
        super(Myembed, self).__init__()
        self.N_freqs = int(N)
        self.freq_bands = 2.**torch.linspace(0., self.N_freqs-1, steps=self.N_freqs)

    def forward(self,x):
        embed_fns=None
        for freq in self.freq_bands:

            if embed_fns is None:
                embed_fns = torch.cat([torch.sin(x * freq * np.pi), torch.cos(x * freq * np.pi)],dim=1)
                # embed_fns = torch.sin(x * freq * np.pi)
            else:
                embed_fns = torch.cat((embed_fns, torch.cat([torch.sin(x * freq * np.pi), torch.cos(x * freq * np.pi)],dim=1)) ,dim=1)
                # embed_fns = torch.cat((embed_fns, torch.sin(x * freq * np.pi)) ,dim=1)
            # out_dim += d

            # print(embed_fns[0,:,0,0])

        return embed_fns

class ScaledLeakyReLUSin(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out_lr = F.leaky_relu(input[:, ::2], negative_slope=self.negative_slope)
        out_sin = torch.sin(input[:, 1::2])
        out = torch.cat([out_lr, out_sin], 1)
        return out * math.sqrt(2)


class StyledResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, blur_kernel=[1, 3, 3, 1], demodulate=True,
                 activation=None, upsample=False, downsample=False):
        super().__init__()

        self.conv1 = StyledConv(in_channel, out_channel, kernel_size, style_dim,
                                demodulate=demodulate, activation=activation)
        self.conv2 = StyledConv(out_channel, out_channel, kernel_size, style_dim,
                                demodulate=demodulate, activation=activation,
                                upsample=upsample, downsample=downsample)

        if downsample or in_channel != out_channel or upsample:
            self.skip = ConvLayer(
                in_channel, out_channel, 1, downsample=downsample, activate=False, bias=False, upsample=upsample,
            )
        else:
            self.skip = None

    def forward(self, input, latent):
        out = self.conv1(input, latent)
        out = self.conv2(out, latent)

        if self.skip is not None:
            skip = self.skip(input)
        else:
            skip = input

        out = (out + skip) / math.sqrt(2)

        return out
