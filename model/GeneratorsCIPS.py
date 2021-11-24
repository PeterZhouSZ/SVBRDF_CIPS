__all__ = ['CIPSskip',
           'CIPSres',
           ]

import math
import os

import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from .blocks import ConstantInput, LFF, StyledConv, ToRGB, PixelNorm, EqualLinear, StyledResBlock, Myembed
from torchvision import transforms

import random

class CIPSskip(nn.Module):
    def __init__(self, img_channels=3, size=256, crop_size = 256, hidden_size=512, n_mlp=8, style_dim=512, lr_mlp=0.01,
                 activation=None, channel_multiplier=2, tileable=False, N_emb=-1, in_pat_path = '', **kwargs):
        super(CIPSskip, self).__init__()
        self.use_pat = True if in_pat_path != '' else False
        self.size = size
        self.in_pat_path = in_pat_path
        demodulate = True
        self.demodulate = demodulate

        if N_emb==-1:
            if self.use_pat:
                # read patterns
                self.in_pats = self._read_patterns()
                c_emb = self.in_pats.shape[0]
            else:
                self.lff = LFF(hidden_size)
                c_emb = hidden_size
        else:
            c_emb = N_emb*4

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
        in_channels = int(self.channels[0])
        print(hidden_size+c_emb, in_channels)
        self.conv1 = StyledConv(int(hidden_size+c_emb), in_channels, 1, style_dim, demodulate=demodulate,
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
        self.crop_size = crop_size
        self.tileable = tileable


        # self.project = project
    def _read_patterns(self):
        patterns = []
        for pat_path in os.listdir(self.in_pat_path):
            if pat_path.split('.')[1] in ['png', 'jpeg', 'jpg']:
                pattern = Image.open(os.path.join(self.in_pat_path, pat_path))

                transform = transforms.Compose(
                    [
                        # transforms.Resize((self.size, self.size)),
                        transforms.ToTensor(),
                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
                    ]
                )

                patterns.append(transform(pattern))

        patterns = torch.cat(patterns, dim=0)
        print(patterns.shape)
        print(patterns[:,0,0])
        return patterns

    def _crop(self, x, tileable=False, center=False):

        b,c,h,w = x.shape
        if center:
            w0 = (w-self.crop_size)*0.5
            h0 = (h-self.crop_size)*0.5
            w0 = int(w0)
            h0 = int(h0)
            # print('center: ', w0, h0)
            if w0+self.crop_size>w or h0+self.crop_size>h:
                raise ValueError('value error of w0')

            return x[:,:,w0:w0+self.size,h0:h0+self.size]

        if not tileable:
            w0 = random.randint(0,w-self.crop_size)
            h0 = random.randint(0,h-self.crop_size)
            if w0+self.crop_size>w or h0+self.crop_size>h:
                raise ValueError('value error of w0')

            return x[:,:,w0:w0+256,h0:h0+256]

        else:
            w0 = random.randint(-self.crop_size,w)
            h0 = random.randint(-self.crop_size,h)

            wc = w0 + self.crop_size
            hc = h0 + self.crop_size

            p = torch.zeros((b,c,self.crop_size,self.crop_size), device='cuda')

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
            elif h0<=0 and (w0<w-self.crop_size and w0>0):
                p[:,:,0:-h0,:] = x[:,:, h+h0:h,w0:wc]
                p[:,:,-h0:,:] = x[:,:, 0:hc, w0:wc]
            # 3
            elif h0<=0 and w0 >=w-self.crop_size:
                p[:,:,0:-h0,0:w-w0] = x[:,:, h+h0:h, w0:w]
                p[:,:,-h0:,0:w-w0] = x[:,:, 0:hc, w0:w]
                p[:,:,0:-h0,w-w0:] = x[:,:, h+h0:h, 0:wc-w]
                p[:,:,-h0:,w-w0:] = x[:,:, 0:hc, 0:wc-w]

            # 4
            elif (h0>0 and h0<h-self.crop_size) and w0<=0:
                p[:,:,:,0:-w0] = x[:,:, h0:hc, w+w0:w]
                p[:,:,:,-w0:] = x[:,:, h0:hc, 0:wc]
            # 5
            elif (h0>0 and h0<h-self.crop_size) and (w0<w-self.crop_size and w0>0):
                p = x[:,:, h0:hc, w0:wc]
            # 6
            elif (h0>0 and h0<h-self.crop_size) and w0 >=w-self.crop_size:
                p[:,:,:,0:w-w0] = x[:,:, h0:hc, w0:w]
                p[:,:,:,w-w0:] = x[:,:, h0:hc, 0:wc-w]

            # 7
            elif h0 >=h-self.crop_size and w0<=0:
                p[:,:,0:h-h0,0:-w0] = x[:,:, h0:h, w+w0:w]
                p[:,:,h-h0:,0:-w0] = x[:,:, 0:hc-h, w+w0:w]
                p[:,:,0:h-h0,-w0:] = x[:,:, h0:h, 0:wc]
                p[:,:,h-h0:,-w0:] = x[:,:, 0:hc-h, 0:wc]
            # 8
            elif h0 >=h-self.crop_size and (w0<w-self.crop_size and w0>0):
                p[:,:,0:h-h0,:] = x[:,:, h0:h,w0:wc]
                p[:,:,h-h0:,:] = x[:,:, 0:hc-h, w0:wc]
            # 9
            elif h0 >=h-self.crop_size and w0 >=w-self.crop_size:
                p[:,:,0:h-h0,0:w-w0] = x[:,:, h0:h, w0:w]
                p[:,:,h-h0:,0:w-w0] = x[:,:, 0:hc-h, w0:w]
                p[:,:,0:h-h0,w-w0:] = x[:,:, h0:h, 0:wc-w]
                p[:,:,h-h0:,w-w0:] = x[:,:, 0:hc-h, 0:wc-w]

            del x

            return p

    def _h_to_N(self, img_in, mode='tangent_space', normal_format='gl', use_input_alpha=False, use_alpha=False, intensity=1.0/3.0, max_intensity=3.0):
        """Atomic function: Normal (https://docs.substance3d.com/sddoc/normal-172825289.html)

        Args:
            img_in (tensor): Input image.
            mode (str, optional): 'tangent space' or 'object space'. Defaults to 'tangent_space'.
            normal_format (str, optional): 'gl' or 'dx'. Defaults to 'dx'.
            use_input_alpha (bool, optional): Use input alpha. Defaults to False.
            use_alpha (bool, optional): Enable the alpha channel. Defaults to False.
            intensity (float, optional): Normalized height map multiplier on dx, dy. Defaults to 1.0/3.0.
            max_intensity (float, optional): Maximum height map multiplier. Defaults to 3.0.

        Returns:
            Tensor: Normal image.
        """
        # grayscale_input_check(img_in, "input height field")
        assert img_in.shape[1]==1, 'should be grayscale image'

        def roll_row(img_in, n):
            return img_in.roll(-n, 2)

        def roll_col(img_in, n):
            return img_in.roll(-n, 3)

        def norm(vec): #[B,C,W,H]
            vec = vec.div(vec.norm(2.0, 1, keepdim=True))
            return vec

        img_size = img_in.shape[2]
        # intensity = intensity * max_intensity * img_size / 256.0 # magic number to match sbs, check it later
        intensity = (intensity * 2.0 - 1.0) * max_intensity * img_size / 256.0 # magic number to match sbs, check it later
        dx = roll_col(img_in, -1) - img_in
        dy = roll_row(img_in, -1) - img_in
        if normal_format == 'gl':
            img_out = torch.cat((intensity*dx, -intensity*dy, torch.ones_like(dx)), 1)
        elif normal_format == 'dx':
            img_out = torch.cat((intensity*dx, intensity*dy, torch.ones_like(dx)), 1)
        else:
            img_out = torch.cat((-intensity*dx, intensity*dy, torch.ones_like(dx)), 1)
        img_out = norm(img_out)
        # if mode == 'tangent_space':
        #     img_out = img_out / 2.0 + 0.5
        
        if use_alpha == True:
            if use_input_alpha:
                img_out = torch.cat([img_out, img_in], dim=1)
            else:
                img_out = torch.cat([img_out, torch.ones(img_out.shape[0], 1, img_out.shape[2], img_out.shape[3])], dim=1)

        return img_out

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
                embed_x=None,
                project=False,
                center_crop=False,
                vis_embed=False,
                ):

        latent = latent[0]

        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)

        if not input_is_latent and not input_is_latent_plus:
            latent = self.style(latent)

        batch_size, _, w, h = coords.shape

        # if not using patterns as input
        if not self.use_pat:
            if embed_x is not None:
                x = embed_x.repeat(batch_size,1,1,1)
            else:
                x = self.lff(coords)

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

            x = torch.cat([x, emb], 1)

            # we do random crop here
            if x.shape[-1]>self.size and (self.training or project):
                print('crop at the top')
                x = self._crop(x, tileable=self.tileable, center=center_crop)

        # use patterns as input
        else:
            x = self.in_pats


        rgb = 0
        latent_plus_list=[]
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

        # print('before final crop:', rgb.shape)
        if rgb.shape[-1]>self.crop_size and (self.training or project):
            # print('crop at the end')
            rgb = self._crop(rgb, tileable=self.tileable, center=center_crop)


        if vis_embed:
            return x_embed


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
