#!/bin/bash


POW_WEIGHT=0
BATCH_SIZE=100
SAVE_FREQ=50
GLR=0.0001
DLR=0.0004
EPOCH=440
SKIP_TYPE='alpha'
VANILLA=''
CLASSES=""
DATA_ROOT="/veu4/santi.pasqual/DB/VCTK_trimmed_train/"
SLICE_SIZE=16384
DPOOL_SLEN=16
STEP_ITERS=1
SAVE_PATH="GSEGAN_liteGEnhancement_RWD-newPASE_nomalign_oldG_hinge"
PASE_CKPT="pase_models/liteGEnhancement_QRNN512_emb100/FE_e29.ckpt"
PASE_CFG="pase_models/liteGEnhancement_QRNN512_emb100/PASE_dense_QRNN512_emb100.cfg"
#Z_DIM=256
NUM_WORKERS=8
BATCH_D="--batch_D"

python -W ignore -u train_gsegan.py --save_path $SAVE_PATH \
	--data_root $DATA_ROOT \
	--batch_size $BATCH_SIZE --save_freq $SAVE_FREQ \
	--epoch $EPOCH \
	--no_train_gen \
	--slice_size $SLICE_SIZE \
	--dpool_slen $DPOOL_SLEN \
	--g_lr $GLR --d_lr $DLR  \
	--pow_weight $POW_WEIGHT --preemph 0 --num_workers $NUM_WORKERS \
	--opt adam --skip_type $SKIP_TYPE \
	--genc_fmaps 64 128 256 512 1024 \
	--gdec_kwidth 31 31 31 31 31 --gdec_fmaps 512 256 128 64 1 \
	--gdec_poolings 4 4 4 4 4 \
	--dnorm_type snorm --partial_snorm \
	--step_iters $STEP_ITERS $BATCH_D \
	--wsegan --dtrans_cfg data/GEnhancement/distortions_SEGANnoises.cfg --rwd \
	--pase_cfg $PASE_CFG \
	--pase_ckpt $PASE_CKPT --gan_loss hinge 

	#--misalign_pair --dnorm_type snorm --partial_snorm \
	#--pase_ckpt pase_models/QRNN/PASE_dense_QRNN.ckpt --ft_fe
#--no_train_gen \
#--noisy_data_root $NOISY_DATA_ROOT


#--dfe_ckpt $FE_CKPT \

#--gfe_ckpt $FE_CKPT \
#--gdec_fmaps 1024 512 256 128 \
