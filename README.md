## CIPS -- Official Pytorch Implementation 

of the paper [Image Generators with Conditionally-Independent Pixel Synthesis](https://arxiv.org/abs/2011.13775)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/image-generators-with-conditionally/image-generation-on-lsun-churches-256-x-256)](https://paperswithcode.com/sota/image-generation-on-lsun-churches-256-x-256?p=image-generators-with-conditionally)

![Teaser](doc/teaser_img.jpg)

## Requirements

pip install -r requirements.txt

## Usage

First create lmdb datasets:

> python prepare_data.py images --out LMDB_PATH --n_worker N_WORKER DATASET_PATH --size 256

This will convert images to jpeg and pre-resizes it. `DATASET_PATH=./Data/ValenTrain`

To train on FFHQ-256 or churches please run:

> python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 train.py --n_sample=8 --batch=4 --fid_batch=8 --Generator=CIPSskip --output_dir=skip-[ffhq/churches] --img2dis --num_workers=16 DATASET_PATH

To train on patches add --crop=PATCH_SIZE. PATCH_SIZE has to be a power of 2.

