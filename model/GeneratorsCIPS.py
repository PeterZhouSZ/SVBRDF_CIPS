__all__ = ['CIPSskip',
           'CIPSres',
           ]

import math

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import ConstantInput, LFF, StyledConv, ToRGB, PixelNorm, EqualLinear, StyledResBlock

import random

class CIPSskip(nn.Module):
    def __init__(self, img_channels=3, size=256, hidden_size=512, n_mlp=8, style_dim=512, lr_mlp=0.01,
                 activation=None, channel_multiplier=2, **kwargs):
        super(CIPSskip, self).__init__()

        self.size = size
        demodulate = True
        self.demodulate = demodulate
        self.lff = LFF(hidden_size)
        self.emb = ConstantInput(hidden_size, size=size)

        self.channels = {
            0: 512,
            1: 512,
            2: 512,
            3: 512,
            4: 256 * channel_multiplier,
            5: 128 * channel_multiplier,
            6: 64 * channel_multiplier,
            7: 32 * channel_multiplier,
            8: 16 * channel_multiplier,
        }

        multiplier = 2
        in_channels = int(self.channels[0])
        self.conv1 = StyledConv(int(multiplier*hidden_size), in_channels, 1, style_dim, demodulate=demodulate,
                                activation=activation)

        self.linears = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.log_size = int(math.log(size, 2))

        self.n_intermediate = self.log_size - 1
        self.to_rgb_stride = 2
        for i in range(0, self.log_size - 1):
            out_channels = self.channels[i]
            self.linears.append(StyledConv(in_channels, out_channels, 1, style_dim,
                                           demodulate=demodulate, activation=activation))
            self.linears.append(StyledConv(out_channels, out_channels, 1, style_dim,
                                           demodulate=demodulate, activation=activation))
            self.to_rgbs.append(ToRGB(out_channels, img_channels, style_dim, upsample=False))

            in_channels = out_channels

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

    def _crop(self, x):

        _,b,w,h = x.shape
        w0 = random.randint(0,w-256)
        h0 = random.randint(0,h-256)
        # print('w0 ',w0)
        # print('h0 ',h0)

        if w0+256>w or h0+256>h:
            raise ValueError('value error of w0')

        return x[:,:,w0:w0+256,h0:h0+256]


    def forward(self,
                coords,
                latent,
                latent_plus=None,
                return_latents=False,
                return_latents_plus=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                input_is_latent_plus=False,
                ):

        latent = latent[0]

        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)

        if not input_is_latent and not input_is_latent_plus:
            latent = self.style(latent)

        # print('coords: ',coords.shape)
        x = self.lff(coords)
        # print('x1: ',x.shape)

        batch_size, _, w, h = coords.shape
        if self.training and w == h == self.size:
            emb = self.emb(x)
            # print('train emb: ',emb.shape)
        else:
            # print('gema...............')
            emb = F.grid_sample(
                self.emb.input.expand(batch_size, -1, -1, -1),
                coords.permute(0, 2, 3, 1).contiguous(),
                padding_mode='border', mode='bilinear',
            )
            # print('gema emb: ',emb.shape)


        x = torch.cat([x, emb], 1)
        # print('x2: ',x.shape)

        rgb = 0

        latent_plus_list=[]

        # we do random crop here
        if x.shape[-1]>256 and self.training:
            x = self._crop(x)
            # print('gema....x.................', x.shape)

        # if input_is_latent_plus:
        #     print('------------------------------',latent[0].shape)

        x, latent_plus = self.conv1(x, latent if not input_is_latent_plus else latent[0],style_plus=input_is_latent_plus)
        # print('+++++++++++++++++++++++++++++++++++',latent_plus.shape)
        latent_index = 1
        latent_plus_list.append(latent_plus)

        for i in range(self.n_intermediate):
            for j in range(self.to_rgb_stride):
                # print('G input_is_latent: ', input_is_latent_plus)
                x, latent_plus = self.linears[i*self.to_rgb_stride + j](x, latent if not input_is_latent_plus else latent[latent_index],style_plus=input_is_latent_plus)
                latent_index += 1
                latent_plus_list.append(latent_plus)

            # print('rgb index', latent_index)
            rgb, latent_plus = self.to_rgbs[i](x, latent if not input_is_latent_plus else latent[latent_index], rgb, style_plus=input_is_latent_plus)
            latent_index += 1
            latent_plus_list.append(latent_plus)


        # print('rgb shape: ', rgb.shape)

        if return_latents:
            return rgb, latent, None

        elif return_latents_plus:
            return rgb, None, latent_plus_list

        else:
            return rgb, None, None


class CIPSres(nn.Module):
    def __init__(self, size=256, hidden_size=512, n_mlp=8, style_dim=512, lr_mlp=0.01,
                 activation=None, channel_multiplier=2, **kwargs):
        super(CIPSres, self).__init__()

        self.size = size
        demodulate = True
        self.demodulate = demodulate
        self.lff = LFF(int(hidden_size))
        self.emb = ConstantInput(hidden_size, size=size)

        self.channels = {
            0: 512,
            1: 512,
            2: 512,
            3: 512,
            4: 256 * channel_multiplier,
            5: 128 * channel_multiplier,
            6: 64 * channel_multiplier,
            7: 64 * channel_multiplier,
            8: 32 * channel_multiplier,
        }

        self.linears = nn.ModuleList()
        in_channels = int(self.channels[0])
        multiplier = 2
        self.linears.append(StyledConv(int(multiplier*hidden_size), in_channels, 1, style_dim, demodulate=demodulate,
                                       activation=activation))

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        for i in range(0, self.log_size - 1):
            out_channels = self.channels[i]
            self.linears.append(StyledResBlock(in_channels, out_channels, 1, style_dim, demodulate=demodulate,
                                               activation=activation))
            in_channels = out_channels

        self.to_rgb_last = ToRGB(in_channels, style_dim, upsample=False)

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

    def forward(self,
                coords,
                latent,
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                ):

        latent = latent[0]

        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)

        if not input_is_latent:
            latent = self.style(latent)

        x = self.lff(coords)

        batch_size, _, w, h = coords.shape
        if self.training and w == h == self.size:
            emb = self.emb(x)
        else:
            emb = F.grid_sample(
                self.emb.input.expand(batch_size, -1, -1, -1),
                coords.permute(0, 2, 3, 1).contiguous(),
                padding_mode='border', mode='bilinear',
            )
        out = torch.cat([x, emb], 1)

        for con in self.linears:
            out = con(out, latent)

        out = self.to_rgb_last(out, latent)

        if return_latents:
            return out, latent
        else:
            return out, None


# rgb index 3
# rgb index 6
# rgb index 9
# rgb index 12
# rgb index 15
# rgb index 18
# rgb index 21
# rgb index 3
# rgb index 6
# rgb index 9
# rgb index 12
# rgb index 15
# rgb index 18
# rgb index 21
# rgb index 3
# rgb index 6
# rgb index 9
# rgb index 12
# rgb index 15
# rgb index 18
# rgb index 21
