## CIPS -- Official Pytorch Implementation 

of the paper [Image Generators with Conditionally-Independent Pixel Synthesis](https://arxiv.org/abs/2011.13775)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/image-generators-with-conditionally/image-generation-on-lsun-churches-256-x-256)](https://paperswithcode.com/sota/image-generation-on-lsun-churches-256-x-256?p=image-generators-with-conditionally)

![Teaser](doc/teaser_img.jpg)

## Requirements

> conda create -n CIPS python=3.7

> conda activate CIPS

> pip install -r requirements.txt

## Usage

1) First create lmdb datasets:

> python prepare_data.py images --out MDB_PATH --n_worker N_WORKER DATASET_PATH --size 256

This will convert images to jpeg and pre-resizes it. `MDB_PATH =./Data/ValenTrain/256`

Or download the MDB file of Valentin Dataset [link](https://drive.google.com/drive/folders/1xFAdBcJiC9KLkPjEC5UkcjEa1OcLTXEk?usp=sharing) and save it to `./Data/ValenTrain/256`

To download bricks dataset, download from this link: [link](https://drive.google.com/file/d/1H1dFjwZUZ1eW_2JUnmW0DwNgHlOYnLBo/view?usp=sharing) and save it to `./Data/`

To download bricks dataset with processed patterns, please download from this link: [link](https://drive.google.com/drive/folders/1O1aw_oEFo9tLVf2RSt_zlpCwmJRv9YpM?usp=sharing)and save it to`./Data/`

2) Training commands:

please check `scripts` folder