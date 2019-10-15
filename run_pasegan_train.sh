#!/bin/bash


POW_WEIGHT=0
BATCH_SIZE=50
SAVE_FREQ=200
GLR=0.0001
DLR=0.0004
EPOCH=165
DATA_ROOT="data/GEnhancement/VCTK_SEGANnoises/train/clean"
NOISY_DATA_ROOT="data/GEnhancement/VCTK_SEGANnoises/train/noisy"
SLICE_SIZE=8000
STEP_ITERS=10
SAVE_PATH="PASEGAN_VCTK_ftfe_ebsz500"
PASE_CFG="pase_models/QRNN512_GEnhancement/PASE_dense_QRNN512.cfg"
PASE_CKPT="pase_models/QRNN512_GEnhancement/FE_e14.ckpt"
NUM_WORKERS=10
D_PRETRAINED_CKPT=""
G_PRETRAINED_CKPT=""

#python -W ignore $HOME/git/line_profiler/kernprof.py -lv train_pasegan.py --save_path $SAVE_PATH \
python -W ignore -u train_pasegan.py --save_path $SAVE_PATH \
	--data_root $DATA_ROOT \
	--batch_size $BATCH_SIZE --save_freq $SAVE_FREQ \
	--epoch $EPOCH \
	--slice_size $SLICE_SIZE \
	--g_lr $GLR --d_lr $DLR  \
	--pow_weight $POW_WEIGHT --preemph 0 --num_workers $NUM_WORKERS \
	--opt adam \
	--step_iters $STEP_ITERS \
	--wsegan \
	--noisy_data_root $NOISY_DATA_ROOT \
	--pase_cfg $PASE_CFG --pase_ckpt $PASE_CKPT --ft_fe

#--no_train_gen

