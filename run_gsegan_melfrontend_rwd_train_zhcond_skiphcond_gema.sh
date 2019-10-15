#!/bin/bash


POW_WEIGHT=0
BATCH_SIZE=64
SAVE_FREQ=500
GLR=0.0001
DLR=0.0004
EPOCH=2000
SKIP_TYPE='alpha'
SKIP_HYPERCOND='--skip_hypercond'
VANILLA=''
CLASSES=""
DATA_ROOT="$HOME/DB/VCTK_trimmed_train/"
SLICE_SIZE=16384
DPOOL_SLEN=16
STEP_ITERS=1
SAVE_PATH="GSEGAN_liteGEnhancement_RWD-melfrontend_nomalign_compressedG_hinge_nopow_zhypercond_skiphypercond_GEMA"
PASE_CKPT="pase_models/liteGEnhancement_QRNN512_emb100/FE_e29.ckpt"
PASE_CFG="pase_models/liteGEnhancement_QRNN512_emb100/PASE_dense_QRNN512_emb100.cfg"
Z_DIM=256
NUM_WORKERS=5
BATCH_D="--batch_D"

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
	--pase_ckpt $PASE_CKPT --gan_loss hinge --z_hypercond $SKIP_HYPERCOND --seed 59  \
	--gema --ema_decay 0.8 --ema_start 0 --cache --frontend_mode mel

