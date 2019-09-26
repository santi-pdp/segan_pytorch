#!/bin/bash


POW_WEIGHT=0.1
BATCH_SIZE=15
SAVE_FREQ=1
GLR=0.0001
DLR=0.0004
EPOCH=400
SKIP_TYPE='alpha'
VANILLA=''
CLASSES=""
DATA_ROOT="data/GEnhancement/VCTK_SEGANnoises/train/clean"
NOISY_DATA_ROOT="data/GEnhancement/VCTK_SEGANnoises/train/noisy"
SLICE_SIZE=16384
DPOOL_SLEN=16
STEP_ITERS=1
SAVE_PATH="tmp_gsegan_p1"
#SAVE_PATH="ckpts_gsegan_publication/GSEGAN_baseline_pow01_400epoch"
#NFFT=2048
DOUTS=277
NFFT=512
NUM_WORKERS=0
#D_PRETRAINED_CKPT="--d_pretrained_ckpt ckpts_gsegan_lsgan/ckpt_gsegan_WSEGAN_onlyadv_epoch150/weights_EOE_D-Discriminator-72300.ckpt"
#G_PRETRAINED_CKPT="--g_pretrained_ckpt ckpts_gsegan_lsgan/ckpt_gsegan_WSEGAN_onlyadv_epoch150/weights_EOE_G-Generator-72300_converted.ckpt"
D_PRETRAINED_CKPT=""
G_PRETRAINED_CKPT=""

python -u train_gsegan.py --save_path $SAVE_PATH \
	--data_root $DATA_ROOT \
	--batch_size $BATCH_SIZE --save_freq $SAVE_FREQ \
	--epoch $EPOCH --no_train_gen \
	--slice_size $SLICE_SIZE \
	--dpool_slen $DPOOL_SLEN \
	--g_lr $GLR --d_lr $DLR  \
	--pow_weight $POW_WEIGHT --preemph 0 --num_workers $NUM_WORKERS \
	--opt adam --skip_type $SKIP_TYPE \
	--genc_fmaps 64 128 256 512 1024 \
	--denc_fmaps 64 128 256 512 1024 \
	--gdec_kwidth 31 31 31 31 31 --gdec_fmaps 512 256 128 64 1 \
	--gdec_poolings 4 4 4 4 4 \
	--misalign_pair --dnorm_type snorm \
	--step_iters $STEP_ITERS \
	--wsegan \
	--noisy_data_root $NOISY_DATA_ROOT


#--dfe_ckpt $FE_CKPT \

#--gfe_ckpt $FE_CKPT \
#--gdec_fmaps 1024 512 256 128 \
