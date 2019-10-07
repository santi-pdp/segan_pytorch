#!/bin/bash


POW_WEIGHT=0.1
BATCH_SIZE=64
SAVE_FREQ=1
GLR=0.0001
DLR=0.0004
EPOCH=2000
SKIP_TYPE='alpha'
VANILLA=''
CLASSES=""
#DATA_ROOT="data/GEnhancement/all_data/train/clean"
#DATA_ROOT="/veu4/santi.pasqual/DB/VCTK_and_LibriSpeech/train/"
DATA_ROOT="/veu4/usuaris26/spascual/DB/VCTK_trimmed_train/"
#DATA_ROOT="/veu4/santi.pasqual/DB/VCTK_trimmed_train/"
#NOISY_DATA_ROOT="data/GEnhancement/all_data/train/noisy"
SLICE_SIZE=16384
DPOOL_SLEN=16
STEP_ITERS=1
#SAVE_PATH="GSEGAN_liteGEnhancement_RWD-ftfe_nomalign_compressedG"
#SAVE_PATH="GSEGAN_liteGEnhancement_RWD_nomalign_compressedG"
#SAVE_PATH="GSEGAN_liteGEnhancement_RWD_nomalign_compressedG_noptPASE"
SAVE_PATH="/veu4/usuaris26/spascual/GSEGAN_liteGEnhancement_RWD-newPASE_nomalign_compressedG_hinge_zhypercond_part2"
#SAVE_PATH="/veu4/usuaris26/spascual/GSEGAN_liteGEnhancement_RWD-newPASE_nomalign_compressedG_hinge_zhypercond_nopow"
#SAVE_PATH="GSEGAN_liteGEnhancement_RWD-newPASE_nomalign_compressedG_hinge_zhypercond_nopow"
#SAVE_PATH="ckpts_gsegan_publication/GSEGAN_baseline_pow01_400epoch"
#NFFT=2048
#PASE_CKPT="pase_models/QRNN/PASE_dense_QRNN.ckpt"
#PASE_CFG="pase_models/QRNN/PASE_dense_QRNN.cfg"
PASE_CKPT="pase_models/liteGEnhancement_QRNN512_emb100/FE_e29.ckpt"
PASE_CFG="pase_models/liteGEnhancement_QRNN512_emb100/PASE_dense_QRNN512_emb100.cfg"
DOUTS=277
NFFT=512
Z_DIM=256
NUM_WORKERS=10
BATCH_D="--batch_D"
#D_PRETRAINED_CKPT="--d_pretrained_ckpt ckpts_gsegan_lsgan/ckpt_gsegan_WSEGAN_onlyadv_epoch150/weights_EOE_D-Discriminator-72300.ckpt"
#G_PRETRAINED_CKPT="--g_pretrained_ckpt ckpts_gsegan_lsgan/ckpt_gsegan_WSEGAN_onlyadv_epoch150/weights_EOE_G-Generator-72300_converted.ckpt"
D_PRETRAINED_CKPT="--d_pretrained_ckpt /veu4/usuaris26/spascual/GSEGAN_liteGEnhancement_RWD-newPASE_nomalign_compressedG_hinge_zhypercond/weights_EOE_D-BaseModel-127259.ckpt"
G_PRETRAINED_CKPT="--g_pretrained_ckpt /veu4/usuaris26/spascual/GSEGAN_liteGEnhancement_RWD-newPASE_nomalign_compressedG_hinge_zhypercond/weights_EOE_G-Generator-127259.ckpt"

python -W ignore -u train_gsegan.py --save_path $SAVE_PATH \
	--data_root $DATA_ROOT \
	--batch_size $BATCH_SIZE --save_freq $SAVE_FREQ \
	--epoch $EPOCH \
	--slice_size $SLICE_SIZE \
	--dpool_slen $DPOOL_SLEN \
	--g_lr $GLR --d_lr $DLR  \
	--pow_weight $POW_WEIGHT --preemph 0 --num_workers $NUM_WORKERS \
	--opt adam --skip_type $SKIP_TYPE \
	--skip_init zero \
	--skip_merge sum \
	--z_dim $Z_DIM \
	--genc_fmaps 64 128 256 256 256 \
	--denc_fmaps 64 128 256 256 256 \
	--gdec_kwidth 31 31 31 31 31 --gdec_fmaps 256 256 128 64 64 \
	--gdec_poolings 4 4 4 4 4 \
	--dnorm_type snorm --partial_snorm $D_PRETRAINED_CKPT $G_PRETRAINED_CKPT \
	--step_iters $STEP_ITERS $BATCH_D \
	--wsegan --dtrans_cfg data/GEnhancement/distortions_SEGANnoises.cfg --rwd \
	--pase_cfg $PASE_CFG \
	--pase_ckpt $PASE_CKPT --gan_loss hinge --z_hypercond --seed 59

	#--misalign_pair --dnorm_type snorm --partial_snorm \
	#--pase_ckpt pase_models/QRNN/PASE_dense_QRNN.ckpt --ft_fe
#--no_train_gen \
#--noisy_data_root $NOISY_DATA_ROOT


#--dfe_ckpt $FE_CKPT \

#--gfe_ckpt $FE_CKPT \
#--gdec_fmaps 1024 512 256 128 \
