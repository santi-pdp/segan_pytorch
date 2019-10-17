#!/bin/bash

CKPT_PATH="GSEGAN_liteGEnhancement_RWD-newPASE_nomalign_compressedG_hinge_nopow_zhypercond_skiphypercond/"

ITER=665972

# please specify the path to your G model checkpoint
# as in weights_G-EOE_<iter>.ckpt
G_PRETRAINED_CKPT="weights_EOE_GEMA-Generator-$ITER".ckpt
SAVE_PREFIX="GEMA"
if [ ! -f $CKPT_PATH/$G_PRETRAINED_CKPT ]; then
	G_PRETRAINED_CKPT="weights_EOE_G-Generator-$ITER".ckpt
	SAVE_PREFIX="G"
fi


# please specify the path to your folder containing
# noisy test files, each wav in there will be processed
TEST_FILES_PATH="$HOME/DB/GEnhancement/CMUArctic/noisy_test"

SEED=900

# please specify the output folder where cleaned files
# will be saved
SAVE_PATH="GSEGAN_CMU_test_zzero"


python -u clean_gsegan.py --g_pretrained_ckpt $CKPT_PATH/$G_PRETRAINED_CKPT \
	--test_dir $TEST_FILES_PATH --cfg_file $CKPT_PATH/train.opts \
	--cuda --synthesis_path $SAVE_PATH --soundfile --seed $SEED --z_zero
