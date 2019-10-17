#!/bin/bash

#CKPT_PATH="ckpts_gsegan_lsgan/ckpt_gsegan_WSEGAN_15epoch/"
#CKPT_PATH="ckpts_gsegan_lsgan/ckpt_gsegan_WSEGAN_interf/"
#CKPT_PATH="ckpts_gsegan_lsgan/ckpt_gsegan_WSEGAN_interf_FE/"
#CKPT_PATH="ckpts_gsegan_lsgan/ckpt_gsegan_WSEGAN_justadv/"
#CKPT_PATH="ckpts_gsegan_lsgan/ckpt_gsegan_WSEGAN_sconv_powloss"
#CKPT_PATH="ckpts_gsegan_lsgan/ckpt_gsegan_WSEGAN_sconv_epoch150"
#CKPT_PATH="ckpts_gsegan_lsgan/ckpt_gsegan_WSEGAN_alpha_epoch150"
#CKPT_PATH="ckpts_gsegan_lsgan/ckpt_gsegan_WSEGAN_onlyadv_epoch150"
#CKPT_PATH="ckpts_gsegan_lsgan/ckpt_gsegan_WSEGAN_sconv_powloss_nofe_epoch150"
#CKPT_PATH="ckpts_gsegan_lsgan/ckpt_gsegan_WSEGAN_alpha_powloss_nofe_epoch150"
#CKPT_PATH="ckpts_gsegan_lsgan/ckpt_gsegan_WSEGAN_Dspkid_alpha_powloss_nofe_epoch150"
#CKPT_PATH="ckpts_gsegan_lsgan/WSEGAN_alpha_slice8192_bsz700_pow01_lessfmaps_nointerf"
#CKPT_PATH="ckpts_gsegan_M274/WSEGAN_alpha_slice8192_bsz700_pow01_lessfmaps_nointerf_noZ"
#CKPT_PATH="ckpts_gsegan_lsgan/ckpt_gsegan_WSEGAN-vanilla_alpha_powloss_nofe_epoch150"
#CKPT_PATH="ckpts_gsegan_lsgan/ckpt_gsegan_WSEGAN_onlyadv_epoch150"
#CKPT_PATH="ckpts_gsegan_publication/WSEGAN_alpha_slice8192_bsz150_nointerf_pow01_sincD"
#CKPT_PATH="ckpts_gsegan_publication/GSEGAN_aco"
#CKPT_PATH="ckpts_gsegan_publication/GSEGAN_baseline"
#CKPT_PATH="ckpts_gsegan_publication/GSEGAN_aco_moneoneLSGAN_cachelabs_ptGD33k"
#CKPT_PATH="ckpts_gsegan_publication/GSEGAN_aco_moneoneLSGAN_cachelabs"
#CKPT_PATH="ckpts_gsegan_publication/GSEGAN_baseline"
#CKPT_PATH="ckpts_gsegan_publication/GSEGAN_aco_moneoneLSGAN_cachelabs_lrs00005"
#CKPT_PATH="ckpts_gsegan_publication/GSEGAN_aco_moneoneLSGAN_cachelabs_ptGD48k_TTUR"
#CKPT_PATH="ckpts_gsegan_publication/GSEGAN_aco_moneoneLSGAN_cachelabs_patGD48k_lrs00005_pow01"
#CKPT_PATH="ckpts_gsegan2/GSEGAN_only-projD2_zhidsum_nolatz_skipsum"
#CKPT_PATH="ckpts_gsegan2/GSEGAN_aco_projD2-paseaco_zhidsum_nolatz_skipsum"
#CKPT_PATH="/veu4/usuaris26/spascual/GSEGAN_liteGEnhancement_RWD-newPASE_nomalign_compressedG_hinge_zhypercond/"
#CKPT_PATH="/veu4/usuaris26/spascual/GSEGAN_liteGEnhancement_RWD-newPASE_nomalign_compressedG_hinge_nopow_zhypercond_skiphypercond_GEMA/"
CKPT_PATH="GSEGAN_liteGEnhancement_RWD-newPASE_nomalign_compressedG_hinge_nopow_zhypercond_skiphypercond/"

#ITER=225094
ITER=665972

# please specify the path to your G model checkpoint
# as in weights_G-EOE_<iter>.ckpt
#G_PRETRAINED_CKPT="weights_EOE_G-Generator-7230.ckpt"
#G_PRETRAINED_CKPT="weights_EOE_G-Generator-9412.ckpt"
#G_PRETRAINED_CKPT="weights_EOE_G-Generator-18100.ckpt"
#G_PRETRAINED_CKPT="weights_EOE_G-Generator-16652.ckpt"
#G_PRETRAINED_CKPT="weights_EOE_G-Generator-25546.ckpt"
#G_PRETRAINED_CKPT="weights_EOE_G-Generator-45612.ckpt"
#G_PRETRAINED_CKPT="weights_EOE_G-Generator-10860.ckpt"
#G_PRETRAINED_CKPT="weights_EOE_G-Generator-61214.ckpt"
#G_PRETRAINED_CKPT="weights_EOE_G-Generator-14460.ckpt"
#G_PRETRAINED_CKPT="weights_EOE_G-Generator-67962.ckpt"
#G_PRETRAINED_CKPT="weights_EOE_G-Generator-13496.ckpt"
#G_PRETRAINED_CKPT="weights_EOE_G-Generator-25476.ckpt"
#G_PRETRAINED_CKPT="weights_EOE_G-Generator-72300_converted.ckpt"
#G_PRETRAINED_CKPT="weights_EOE_G-Generator-96500.ckpt"
#G_PRETRAINED_CKPT="weights_EOE_G-Generator-11086.ckpt"
#G_PRETRAINED_CKPT="weights_EOE_G-Generator-33258.pt_ckpt"
#G_PRETRAINED_CKPT="weights_EOE_G-Generator-4338.ckpt"
#G_PRETRAINED_CKPT="weights_EOE_G-Generator-17352.ckpt"
#G_PRETRAINED_CKPT="weights_EOE_G-Generator-48200.ckpt"
#G_PRETRAINED_CKPT="weights_EOE_G-Generator-18316.ckpt"
G_PRETRAINED_CKPT="weights_EOE_GEMA-Generator-$ITER".ckpt
SAVE_PREFIX="GEMA"
if [ ! -f $CKPT_PATH/$G_PRETRAINED_CKPT ]; then
	G_PRETRAINED_CKPT="weights_EOE_G-Generator-$ITER".ckpt
	SAVE_PREFIX="G"
fi


# please specify the path to your folder containing
# noisy test files, each wav in there will be processed
#TEST_FILES_PATH="data_gsegan/distorted6_testset_trimsil"
#TEST_FILES_PATH="$HOME/DB/MiniCHiME5_5h_ct/train"
#TEST_FILES_PATH="$HOME/DB/TIMIT_rev_noise/train"
#TEST_FILES_PATH="$HOME/DB/chime5segmented/"
TEST_FILES_PATH="$HOME/DB/GEnhancement/CMUArctic/noisy_test"
#TEST_FILES_PATH="data_gsegan/distorted_testset_trimsil_M274"

SEED=900

# please specify the output folder where cleaned files
# will be saved
#SAVE_PATH="synth_gsegan_15epoch"
#SAVE_PATH="synth_gsegan_interf"
#SAVE_PATH="synth_gsegan_interf_FE"
#SAVE_PATH="synth_gsegan_justadv"
#SAVE_PATH="synth_gsegan_sconv_powloss"
#SAVE_PATH="synth_gsegan_sconv_powloss_epoch150_18860_seed$SEED"
#SAVE_PATH="synth_gsegan_alpha_epoch150_45612_seed$SEED"
#SAVE_PATH="synth_gsegan_onlyadv_epoch150_25546_seed$SEED"
#SAVE_PATH="synth_gsegan_onlyadv_epoch150_61214_seed$SEED"
#SAVE_PATH="synth_gsegan_sconvpowlossnofe_epoch150_14460_seed$SEED"
#SAVE_PATH="synth_gsegan_alphapowlossnofe_epoch150_67962_seed$SEED"
#SAVE_PATH="synth_gsegan_Dspkid_6647_bsz700_slice8192_seed$SEED"
#SAVE_PATH="synth_gsegan_M274_25476_seed$SEED"
#SAVE_PATH="synth_gsegan_M274_allmodel_72300_seed$SEED"
#SAVE_PATH="synth_gsegan_onlyadv-72300_seed$SEED"
#SAVE_PATH="synth_gsegan_sconv-pow01-nofe-72300_seed$SEED"
#SAVE_PATH="synth_gsegan_sincD-slice8192-96500_seed$SEED"
#SAVE_PATH="synth_gsegan_aco-11086_seed$SEED"
#SAVE_PATH="synth_gsegan_aco_ptGD33k-17352_seed$SEED"
#SAVE_PATH="synth_gsegan_aco_lrs00005-48200_seed$SEED"
#SAVE_PATH="synth_gsegan_aco_ptGD48k-TTUR-18316_seed$SEED"
#SAVE_PATH="synth_gsegan2_$ITER"_"seed$SEED"
#SAVE_PATH="synth_gsegan2aco-paseaco_$ITER"_"seed$SEED"
#SAVE_PATH="$HOME/DB/MiniCHiME5_5h_ct/train_enhanced_GSEGAN-iter"$ITER"_hinge_zhypercond_zzero"
#SAVE_PATH="$HOME/DB/chime5segmented_enhanced/"$SAVE_PREFIX"-iter"$ITER"_hinge_nopow_zhypercond_skiphypercond_zzero"
SAVE_PATH="GSEGAN_CMU_test_zzero"


python -u clean_gsegan.py --g_pretrained_ckpt $CKPT_PATH/$G_PRETRAINED_CKPT \
	--test_dir $TEST_FILES_PATH --cfg_file $CKPT_PATH/train.opts \
	--cuda --synthesis_path $SAVE_PATH --soundfile --seed $SEED --z_zero
