__all__ = ['CIPSskip',
           'CIPSres',
           ]

import math
import os

import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from .blocks import ConstantInput, LFF, StyledConv, ToRGB, PixelNorm, EqualLinear, StyledResBlock, Myembed, mycrop, PatembNet
from torchvision import transforms

import random

class CIPSskip(nn.Module):
    def __init__(self, img_channels=3, size=256, crop_size = 256, hidden_size=512, n_mlp=8, style_dim=512, lr_mlp=0.01,
                 activation=None, channel_multiplier=2, tileable=False, N_emb=-1, in_pat=None, in_pat_c=0, emb_pat='net_conv', add_chconv=False, **kwargs):
        super(CIPSskip, self).__init__()

        self.size = size
        demodulate = True
        self.demodulate = demodulate
        self.in_pat = in_pat
        self.in_pat_c = in_pat_c
        self.emb_pat = emb_pat
        self.add_chconv = add_chconv
        print('add channel conv: ', self.add_chconv)

        if 'net' in self.emb_pat:
            self.patnet = PatembNet(emb_pat,self.in_pat_c)
            # self.in_pat_c = 32 # hard code emb_pat if use network 

        if N_emb==-1:
            self.lff = LFF(hidden_size)
            c_emb = hidden_size
        else:
            c_emb = N_emb*4

        if self.emb_pat=='emb_blur':
            self.pat_emb = ConstantInput(hidden_size, size=size)
            c_emb+=hidden_size
        elif self.emb_pat=='ff_blur':
            c_emb +=N_emb*8

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

        # multiplier = 2
        inc = int(hidden_size+c_emb)
        if in_pat is None:
            assert in_pat_c==0, 'pat channel must be 0 if NOT use in_pat'
        elif in_pat == 'top':
            if self.emb_pat!='ff_blur':
                inc += self.in_pat_c

        in_channels = int(self.channels[0])
        print('1st layer: ',inc, in_channels)

        self.conv1 = StyledConv(inc, in_channels, 1, style_dim, demodulate=demodulate,
                                activation=activation)

        self.linears = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.log_size = int(math.log(size, 2))

        self.n_intermediate = self.log_size - 1
        self.to_rgb_stride = 2 if not self.add_chconv else 3
        for i in range(0, self.log_size - 1):
            out_channels = self.channels[i]

            # add input patterns all layers
            if in_pat=='all':
                print('all: ', in_channels + self.in_pat_c, 'all2: ', out_channels + self.in_pat_c)
                self.linears.append(StyledConv(in_channels + self.in_pat_c, out_channels, 1, style_dim,
                                               demodulate=demodulate, activation=activation))
                self.linears.append(StyledConv(out_channels + self.in_pat_c, out_channels, 1, style_dim,
                                               demodulate=demodulate, activation=activation))   

            else:
                print('all: ', in_channels, 'all2: ', out_channels)
                self.linears.append(StyledConv(in_channels, out_channels, 1, style_dim,
                                               demodulate=demodulate, activation=activation))

                # add channel conv
                if self.add_chconv:
                    self.linears.append(StyledConv(out_channels, out_channels, 5, style_dim,
                                                   demodulate=demodulate, activation=activation, channel_conv=True))

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
        self.crop_size = crop_size
        self.tileable = tileable

    def forward(self,
                coords,
                latent,
                rand0=None,
                in_pats=None,
                latent_plus=None,
                return_latents=False,
                return_latents_plus=False,
                input_is_latent=False,
                input_is_latent_plus=False,
                embed=None,
                crop=False,
                center_crop=False,
                ):

        
        batch_size, _, w, h = coords.shape

        latent = self.style(latent[0]) if not input_is_latent and not input_is_latent_plus else latent[0]
        x = embed.repeat(batch_size,1,1,1) if embed is not None else self.lff(coords)

        if self.training and w == h == self.size:
            emb = self.emb(x)
        else:
            # print('coords', coords.shape)
            emb = F.grid_sample(
                self.emb.input.expand(batch_size, -1, -1, -1),
                coords.permute(0, 2, 3, 1).contiguous(),
                padding_mode='border', mode='bilinear',
            )

        x = torch.cat([x, emb], 1)

        # crop coords
        if x.shape[-1]>self.size and (self.training or crop):
            x = mycrop(x, size=self.crop_size, tileable=self.tileable, center=center_crop, rand0=rand0)

        # pat emb net
        # if 'net' in self.emb_pat:
        #     in_pats = self.patnet(in_pats)

        # # add pat embeddings
        # if self.emb_pat=='emb_blur':
        #     if self.training and w == h == self.size:
        #         pat_emb = self.pat_emb(in_pats)
        #     else:
        #         print('in_pats', in_pats.shape)
        #         pat_emb = F.grid_sample(
        #             self.pat_emb.input.expand(batch_size, -1, -1, -1),
        #             coords.permute(0, 2, 3, 1).contiguous(),
        #             padding_mode='border', mode='bilinear',
        #         )

        #     in_pats = torch.cat([in_pats, pat_emb], 1)

        assert x.shape[-1]==in_pats.shape[-1],f'shape of x {x.shape[-1]} and in_pat {in_pats.shape[-1]} not match'

        # if pattern to top
        if self.in_pat=='top':
            x = torch.cat([x, in_pats],1)

        rgb = 0
        latent_plus_list=[]
        # print('x: ', x.shape)
        x, latent_plus = self.conv1(x, latent if not input_is_latent_plus else latent[0],style_plus=input_is_latent_plus)
        # print('+++++++++++++++++++++++++++++++++++',latent_plus.shape)
        latent_index = 1
        latent_plus_list.append(latent_plus)

        for i in range(self.n_intermediate):
            for j in range(self.to_rgb_stride):
                # print('G input_is_latent: ', input_is_latent_plus)
                x = torch.cat([x,in_pats],1) if self.in_pat=='all' else x
                # print('all in shape (with pattern):', x.shape)

                x, latent_plus = self.linears[i*self.to_rgb_stride + j](x, latent if not input_is_latent_plus else latent[latent_index],style_plus=input_is_latent_plus)
                latent_index += 1
                latent_plus_list.append(latent_plus)

            # print('rgb index', latent_index)
            rgb, latent_plus = self.to_rgbs[i](x, latent if not input_is_latent_plus else latent[latent_index], rgb, style_plus=input_is_latent_plus)
            latent_index += 1
            latent_plus_list.append(latent_plus)

        # if rgb.shape[-1]>self.crop_size and (self.training or crop):
        #     print('crop at the end')
        #     rgb = self._crop(rgb, tileable=self.tileable, center=center_crop)


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
