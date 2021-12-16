import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import model
from dataset import MultiScaleDataset, ImageDataset
# from calculate_fid import calculate_fid
from distributed import get_rank, synchronize, reduce_loss_dict
from tensor_transforms import convert_to_coord_format

from model.blocks import mycrop

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    r1_loss = torch.tensor(0.0, device=device)
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
    else:
        g_module = generator
        d_module = discriminator
        
    accum = 0.5 ** (32 / (10 * 1000))

    converted = convert_to_coord_format(args.batch, args.coords_size, args.coords_size, integer_values=False)
    embed_xy = None
    if args.N_emb>0:
        from model.blocks import Myembed
        embed_fn = Myembed(args.N_emb)
        print('---------converted ------', converted.shape)
        embed_xy = embed_fn(converted[0:1,...])
        print('---------NERF embedding ------', embed_xy.shape)


    real_pat_c = 1 if 'net' in args.emb_pat else args.in_pat_c

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print('Done!')
            break

        load_img = next(loader) 
        in_pats = load_img[:, 0:real_pat_c, :, :].to(device) # [B, load_img_size, H, W]
        # print('in_pats:', in_pats.shape)

        # crop images if needed
        rand0 = None
        if not args.no_tileable:
            rand0 = [random.randint(-args.crop_size, load_img.shape[-1]),random.randint(-args.crop_size, load_img.shape[-1])]
        else:
            rand0 = [random.randint(0,load_img.shape[-1]-args.crop_size), random.randint(0,load_img.shape[-1]-args.crop_size)]

        # crop real image
        real_img = mycrop(load_img[:, -5:, :, :], size=args.crop_size, tileable=not args.no_tileable, rand0=rand0) if load_img.shape[-1] > args.crop_size else load_img[:, -5:, :, :]

        # print('real_img:', real_img.shape)

        real_img = real_img.to(device) # [B, crop_size, H, W]
        converted = converted.to(device)
        if embed_xy is not None:
            embed_xy = embed_xy.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)

        fake_img, _, _, in_pats_cr = generator(converted, noise, rand0=rand0, embed = embed_xy, in_pats = in_pats)

        # print('fake image:', fake_img.shape)
        key = np.random.randint(n_scales)

        fake = fake_img if args.img2dis else torch.cat([fake_img, converted], 1)
        fake_pred = discriminator(fake, key, in_pats = in_pats_cr if not args.no_pat_D else None)

        real = real_img if args.img2dis else real_stack
        real_pred = discriminator(real, key, in_pats = in_pats_cr if not args.no_pat_D else None)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict['d'] = d_loss
        loss_dict['real_score'] = real_pred.mean()
        loss_dict['fake_score'] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real.requires_grad = True
            real_pred = discriminator(real, key, in_pats = in_pats_cr if not args.no_pat_D else None)
            r1_loss = d_r1_loss(real_pred, real)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict['r1'] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)

        fake_img, _, _, in_pats_cr = generator(converted, noise, rand0=rand0, embed = embed_xy, in_pats = in_pats)
        fake = fake_img if args.img2dis else torch.cat([fake_img, converted], 1)
        fake_pred = discriminator(fake, key, in_pats = in_pats_cr if not args.no_pat_D else None)
        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict['g'] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        loss_dict['path'] = path_loss
        loss_dict['path_length'] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced['d'].mean().item()
        g_loss_val = loss_reduced['g'].mean().item()
        r1_val = loss_reduced['r1'].mean().item()
        path_loss_val = loss_reduced['path'].mean().item()
        real_score_val = loss_reduced['real_score'].mean().item()
        fake_score_val = loss_reduced['fake_score'].mean().item()
        path_length_val = loss_reduced['path_length'].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f'd: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; '
                    f'path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}'
                )
            )

            if i % 100 == 0:
                writer.add_scalar("Generator", g_loss_val, i)
                writer.add_scalar("Discriminator", d_loss_val, i)
                writer.add_scalar("R1", r1_val, i)
                writer.add_scalar("Path Length Regularization", path_loss_val, i)
                writer.add_scalar("Mean Path Length", mean_path_length, i)
                writer.add_scalar("Real Score", real_score_val, i)
                writer.add_scalar("Fake Score", fake_score_val, i)
                writer.add_scalar("Path Length", path_length_val, i)

            if i % 1000 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    converted_full = convert_to_coord_format(args.n_sample, args.coords_size, args.coords_size, device,
                                                             integer_values=args.coords_integer_values)
                    if args.generate_by_one:
                        converted_full = convert_to_coord_format(1, args.coords_size, args.coords_size, device,
                                                                 integer_values=args.coords_integer_values)
                        samples = []
                        for sz in sample_z:
                            sample, _, _,_ = g_ema(converted_full, [sz.unsqueeze(0)], embed = embed_xy, in_pats = in_pats)
                            samples.append(sample)
                        sample = torch.cat(samples, 0)
                    else:
                        sample_z = torch.randn(args.n_sample, args.latent, device=device)
                        sample, _, _,_ = g_ema(converted_full, [sample_z], embed = embed_xy, in_pats = in_pats)
                        del sample_z

                    if sample.shape[1]==10:

                        # save N
                        utils.save_image(
                            sample[:,0:3,:,:],
                            os.path.join(path, 'outputs', args.output_dir, 'images', f'{str(i).zfill(6)}_N.png'),
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )
                        # save D
                        utils.save_image(
                            sample[:,3:6,:,:],
                            os.path.join(path, 'outputs', args.output_dir, 'images', f'{str(i).zfill(6)}_D.png'),
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )
                        # save R
                        utils.save_image(
                            sample[:,6:7,:,:].repeat(1,3,1,1),
                            os.path.join(path, 'outputs', args.output_dir, 'images', f'{str(i).zfill(6)}_R.png'),
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )
                        # save S
                        utils.save_image(
                            sample[:,7:10,:,:],
                            os.path.join(path, 'outputs', args.output_dir, 'images', f'{str(i).zfill(6)}_S.png'),
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )
                        
                        # save N
                        utils.save_image(
                            real_img[:,0:3,:,:],
                            os.path.join(path,f'outputs/{args.output_dir}/images/real_patch_{str(key)}_{str(i).zfill(6)}_N.png'),
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )
                        # save D
                        utils.save_image(
                            real_img[:,3:6,:,:],
                            os.path.join(path,f'outputs/{args.output_dir}/images/real_patch_{str(key)}_{str(i).zfill(6)}_D.png'),
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )
                        # save R
                        utils.save_image(
                            real_img[:,6:7,:,:].repeat(1,3,1,1),
                            os.path.join(path,f'outputs/{args.output_dir}/images/real_patch_{str(key)}_{str(i).zfill(6)}_R.png'),
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )
                        # save S
                        utils.save_image(
                            real_img[:,7:10,:,:],
                            os.path.join(path,f'outputs/{args.output_dir}/images/real_patch_{str(key)}_{str(i).zfill(6)}_S.png'),
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )

                    elif sample.shape[1]==5:

                        ######## save for test images
                        if args.in_pat_c != 0:
                            utils.save_image(
                                in_pats[:,0:1,:,:].repeat(1,3,1,1),
                                os.path.join(path, 'outputs', args.output_dir, 'images', f'{str(i).zfill(6)}_Pat_e.png'),
                                nrow=int(args.n_sample ** 0.5),
                                normalize=True,
                                range=(-1, 1),
                            )

                        # save H
                        utils.save_image(
                            sample[:,0:1,:,:].repeat(1,3,1,1),
                            os.path.join(path, 'outputs', args.output_dir, 'images', f'{str(i).zfill(6)}_H_e.png'),
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )
                        # save D
                        utils.save_image(
                            ((sample[:,1:4,:,:]+1)*0.5)**(1/2.2),
                            os.path.join(path, 'outputs', args.output_dir, 'images', f'{str(i).zfill(6)}_D_e.png'),
                            nrow=int(args.n_sample ** 0.5),
                            # normalize=True,
                            # range=(-1, 1),
                        )
                        # save R
                        utils.save_image(
                            sample[:,4:5,:,:].repeat(1,3,1,1),
                            os.path.join(path, 'outputs', args.output_dir, 'images', f'{str(i).zfill(6)}_R_e.png'),
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )

                        # save H
                        utils.save_image(
                            real_img[:,0:1,:,:].repeat(1,3,1,1),
                            os.path.join(path,f'outputs/{args.output_dir}/images/real_patch_{str(key)}_{str(i).zfill(6)}_H.png'),
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )
                        # save D
                        utils.save_image(
                            ((real_img[:,1:4,:,:]+1)*0.5)**(1/2.2),
                            os.path.join(path,f'outputs/{args.output_dir}/images/real_patch_{str(key)}_{str(i).zfill(6)}_D.png'),
                            nrow=int(args.n_sample ** 0.5),
                            # normalize=True,
                            # range=(-1, 1),
                        )
                        # save R
                        utils.save_image(
                            real_img[:,4:5,:,:].repeat(1,3,1,1),
                            os.path.join(path,f'outputs/{args.output_dir}/images/real_patch_{str(key)}_{str(i).zfill(6)}_R.png'),
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )


                        ##### save for training
                        if args.in_pat_c != 0:
                            # for k in range(in_pats_cr.shape[1]):
                            utils.save_image(
                                in_pats_cr[:,0:1,:,:].repeat(1,3,1,1),
                                os.path.join(path, 'outputs', args.output_dir, 'images', f'{str(i).zfill(6)}_Pat_tr.png'),
                                nrow=int(args.n_sample ** 0.5),
                                normalize=True,
                                range=(-1, 1),
                            )

                        # save H
                        utils.save_image(
                            fake_img[:,0:1,:,:].repeat(1,3,1,1),
                            os.path.join(path, 'outputs', args.output_dir, 'images', f'{str(i).zfill(6)}_H_tr.png'),
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )
                        # save D
                        utils.save_image(
                            ((fake_img[:,1:4,:,:]+1)*0.5)**(1/2.2),
                            os.path.join(path, 'outputs', args.output_dir, 'images', f'{str(i).zfill(6)}_D_tr.png'),
                            nrow=int(args.n_sample ** 0.5),
                            # normalize=True,
                            # range=(-1, 1),
                        )
                        # save R
                        utils.save_image(
                            fake_img[:,4:5,:,:].repeat(1,3,1,1),
                            os.path.join(path, 'outputs', args.output_dir, 'images', f'{str(i).zfill(6)}_R_tr.png'),
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )

                    else:
                        utils.save_image(
                            sample,
                            os.path.join(path, 'outputs', args.output_dir, 'images', f'{str(i).zfill(6)}.png'),
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )

                        utils.save_image(
                            real_img,
                            os.path.join(
                                path,
                                f'outputs/{args.output_dir}/images/real_patch_{str(key)}_{str(i).zfill(6)}.png'),
                            nrow=int(real_img.size(0) ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )

            if i % args.save_checkpoint_frequency == 0:
                torch.save(
                    {
                        'g': g_module.state_dict(),
                        'd': d_module.state_dict(),
                        'g_ema': g_ema.state_dict(),
                        'g_optim': g_optim.state_dict(),
                        'd_optim': d_optim.state_dict(),
                    },
                    os.path.join(
                        path,
                        f'outputs/{args.output_dir}/checkpoints/{str(i).zfill(6)}.pt'),
                )

                torch.save(
                    {
                        'g_ema': g_ema.state_dict(),
                    },
                    os.path.join(
                        path,
                        f'outputs/{args.output_dir}/checkpoints_eval/{str(i).zfill(6)}.pt'),
                )


def filter_args(args):
    if args.in_pat is not None:
        args.n_sample=args.batch


    return args



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--out_path', type=str, default='.')
    parser.add_argument('--device', type=str, default='cuda')

    # fid
    parser.add_argument('--fid_samples', type=int, default=50000)
    parser.add_argument('--fid_batch', type=int, default=16)

    # training
    parser.add_argument('--iter', type=int, default=1200000)
    parser.add_argument('--n_sample', type=int, default=64)
    parser.add_argument('--N_emb', type=int, default=-1, help='Number of emb channels: -1: LFF | 0 | >0: NERF')
    parser.add_argument('--generate_by_one', action='store_true')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--pat_lr', type=float, default=0.002)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--save_checkpoint_frequency', type=int, default=20000)
    parser.add_argument('--no_tileable', action='store_true')
    parser.add_argument('--in_pat', type=str, default=None, help='None (default): no pattern || top: pattern into top || all: pattern into all layers')
    parser.add_argument('--emb_pat', type=str, default='blur', help='blur (default): blurred patterns || net: network learned patterns')

    # dataset
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--resize_data', action='store_true')
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--coords_size', type=int, default=256)

    # Generator params
    parser.add_argument('--Generator', type=str, default='ModSIREN')
    parser.add_argument('--coords_integer_values', action='store_true')
    parser.add_argument('--img_c', type=int, default=3, help='# of output imag channels')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--fc_dim', type=int, default=512)
    parser.add_argument('--latent', type=int, default=512)
    parser.add_argument('--activation', type=str, default=None)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--mixing', type=float, default=0.)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--in_pat_c', type=int, default=0, help='# of input channels')
    parser.add_argument('--pat_index', type=int, default=-1, help='-1: use all 8 patterns || 0: use binary pattern || 1: use distance transform pattern')
    parser.add_argument('--add_chconv', action='store_true')

    # Discriminator params
    parser.add_argument('--Discriminator', type=str, default='Discriminator')
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--r1', type=float, default=10)
    parser.add_argument('--img2dis',  action='store_true')
    parser.add_argument('--no_pat_D',  action='store_true')
    parser.add_argument('--n_first_layers', type=int, default=0)

    args = parser.parse_args()
    path = args.out_path

    device = args.device

    filter_args(args)

    print(f'coords_in: {args.coords_size} --> crop at beginning is {args.coords_size>args.size} --> generator_size: {args.size} -->  crop at end is {args.size>args.crop_size} --> real_size: {args.crop_size} (use resize: {args.resize_data})')
    # print('device', device)

    Generator = getattr(model, args.Generator)
    Discriminator = getattr(model, args.Discriminator)
    # print('Discriminator', Discriminator)

    os.makedirs(os.path.join(path, 'outputs', args.output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(path, 'outputs', args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(path, 'outputs', args.output_dir, 'checkpoints_eval'), exist_ok=True)
    args.logdir = os.path.join(path, 'tensorboard', args.output_dir)
    os.makedirs(args.logdir, exist_ok=True)
    # args.path_fid = os.path.join(path, 'fid', args.output_dir)
    # os.makedirs(args.path_fid, exist_ok=True)

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    args.n_mlp = 8
    args.dis_input_c = args.img_c if args.img2dis else args.img_c+2
    # print('img2dis', args.img2dis, 'dis_input_size', args.dis_input_c)

    args.start_iter = 0
    n_scales = int(math.log(args.size//args.crop_size, 2)) + 1

    print('self.in_pat_c ',args.in_pat_c)
    generator = Generator(img_channels = args.img_c, size=args.size, crop_size=args.crop_size, hidden_size=args.fc_dim, style_dim=args.latent, n_mlp=args.n_mlp,
                          activation=args.activation, channel_multiplier=args.channel_multiplier, tileable=not args.no_tileable, N_emb=args.N_emb, 
                          in_pat=args.in_pat, in_pat_c = args.in_pat_c, emb_pat = args.emb_pat, add_chconv=args.add_chconv,
                          ).to(device)

    # for param in generator.named_parameters():
    #     if 'patnet' in param[0]:
    #         print(param[0], param[1].shape)

    print('generator N params', sum(p.numel() for p in generator.parameters() if p.requires_grad))

    discriminator = Discriminator(
        size=args.crop_size, channel_multiplier=args.channel_multiplier, n_scales=n_scales, input_size=args.dis_input_c, n_first_layers=args.n_first_layers,
        in_pat=args.in_pat if not args.no_pat_D else None, in_pat_c = args.in_pat_c, emb_pat = args.emb_pat, hidden_size=args.fc_dim,
    ).to(device)

    g_ema = Generator(img_channels = args.img_c, size=args.size, crop_size=args.crop_size, hidden_size=args.fc_dim, style_dim=args.latent, n_mlp=args.n_mlp,
                      activation=args.activation, channel_multiplier=args.channel_multiplier, tileable=not args.no_tileable, N_emb=args.N_emb, 
                      in_pat=args.in_pat, in_pat_c = args.in_pat_c, emb_pat = args.emb_pat, add_chconv=args.add_chconv,
                      ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)
    # print('Discriminator', discriminator)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)


    if 'net' not in args.emb_pat:
        g_optim = optim.Adam(
            generator.parameters(),
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )
    else:
        print(f'....................................use two learning rate for genertor and generator.embpat. generator {args.lr} || patnet {args.pat_lr} ')
        g_optim = optim.Adam(
            [
            {'params':[param[1] for param in generator.named_parameters() if 'patnet' not in param[0]],},
            {'params':generator.patnet.parameters(), 'lr':args.pat_lr*g_reg_ratio}
            ],
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )



    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print('load model:', args.ckpt)

        ckpt = torch.load(args.ckpt)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt['g'])
        discriminator.load_state_dict(ckpt['d'])
        g_ema.load_state_dict(ckpt['g_ema'])

        g_optim.load_state_dict(ckpt['g_optim'])
        d_optim.load_state_dict(ckpt['d_optim'])

        del ckpt
        torch.cuda.empty_cache()

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    # transform = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #     ]
    # )
    # transform_fid = transforms.Compose([
    #                                    transforms.ToTensor(),
    #                                    transforms.Lambda(lambda x: x.mul_(255.).byte())])

    dataset = MultiScaleDataset(args.path, integer_values=args.coords_integer_values, pat_c = args.in_pat_c, pat_index = args.pat_index, emb_pat=args.emb_pat, resize=args.resize_data)
    # fid_dataset = ImageDataset(args.path, transform=transform_fid, resolution=args.coords_size, to_crop=args.to_crop)
    # fid_dataset.length = args.fid_samples
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    writer = SummaryWriter(log_dir=args.logdir)

    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device)
