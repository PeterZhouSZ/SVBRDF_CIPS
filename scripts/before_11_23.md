

## Usage

1) First create lmdb datasets:

> python prepare_data.py images --out MDB_PATH --n_worker N_WORKER DATASET_PATH --size 256

This will convert images to jpeg and pre-resizes it. `MDB_PATH =./Data/ValenTrain/256`

Or download the MDB file of Valentin Dataset [link](https://drive.google.com/drive/folders/1xFAdBcJiC9KLkPjEC5UkcjEa1OcLTXEk?usp=sharing) and save it to `./Data/ValenTrain/256`

2) Training commands:

a. To train on SVBRDF dataset please run:

> python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 train.py --n_sample=16 --batch=16 --fid_batch=8 --Generator=CIPSskip --output_dir=ValenTrain --img2dis --num_workers=16 DATASET_PATH --img_c 10

where `DATASET_PATH=./Data/ValenTrain/256`. To train on patches add --crop=PATCH_SIZE. PATCH_SIZE has to be a power of 2.


b. To run on 512x512 with random crop and make it tileable, here is command:

> python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 train.py --n_sample=4 --batch=16 --fid_batch=8 --Generator=CIPSskip --output_dir=ValenTrain512_tile_Nemb10 --img2dis --num_workers=16 DATASET_PATH --img_c 10 --coords_size 512 --tileable --N_emb 10

c. To run on 512x512 with random crop `at the end` and make it tileable, here is command (note that the size and coords_size may need to be adjusted to fit the memory):

> python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 train.py --n_sample=2 --batch=1 --fid_batch=8 --Generator=CIPSskip --output_dir=ValenTrain512e_tile_Nemb10 --img2dis --num_workers=16 DATASET_PATH --img_c 10 --coords_size 512 --size 512 --tileable --N_emb 10

d. To run on 512x512 with random crop `at the end`, make it tileable, use `height map` as well

> python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 train.py --n_sample=2 --batch=1 --fid_batch=8 --Generator=CIPSskip --output_dir=ValenTrain512e_tile_Nemb10_H --img2dis --num_workers=16 DATASET_PATH --img_c 8 --coords_size 512 --size 512 --tileable --N_emb 10 --use_height

e. To run on 512x512 with random crop `at the beginning`, make it tileable, use `height map` as well

> python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 train.py --n_sample=2 --batch=2 --fid_batch=8 --Generator=CIPSskip --output_dir=ValenTrain512_tile_Nemb10_H --img2dis --num_workers=16 DATASET_PATH --img_c 8 --coords_size 512 --tileable --N_emb 10 --use_height
