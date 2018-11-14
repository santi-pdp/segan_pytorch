# Speech Enhancement Generative Adversarial Network in PyTorch

Two models are ready to train and use to make wav2wav speech enhancement conversions. SEGAN+ is an
improved version of SEGAN [1], denoising utterances with its generator network (G). To train this model,
the following command should be ran:

```
python train.py --save_path ckpt_segan+ --batch_size 300 \
		--clean_trainset data/clean_trainset \
		--noisy_trainset data/noisy_trainset \
		--cache_dir data/cache
```

Read `run_segan+_train.sh` for more guidance. This will use the default parameters to structure both G and D, but they can be tunned with many options. For example, one can play with `--d_pretrained_ckpt` and/or `--g_pretrained_ckpt` to specify a departure pre-train checkpoint to fine-tune some characteristics of our enhancement system, like language, as in [2].

Cleaning files is done by specifying the generator weights checkpoint, its config file from training and appropriate paths for input and output files (Use `soundfile` wav writer backend (recommended) specifying the `--soundfile` flag):

```
python clean.py --g_pretrained_ckpt ckpt_segan+/<weights_ckpt_for_G> \
		--cfg_file ckpt_segan+/train.opts --synthesis_path enhanced_results \
		--test_files data/noisy_testset --soundfile
```

Read `run_segan+_clean.sh` for more guidance.

There is a WSEGAN, which stands for the dewhispering SEGAN [3]. This system is activated (rather than vanilla SEGAN) by specifying the `--wsegan` flag.

### Disclaimer:

* Multi-GPU is not supported yet in this framework.
* Virtual Batch Norm is not included as in the very first SEGAN code, as similar results to those of original paper can be obtained with regular BatchNorm in D (ONLY D).

### References:

1. [SEGAN: Speech Enhancement Generative Adversarial Network (Pascual et al. 2017)](https://arxiv.org/abs/1703.09452)
2. [Language and Noise Transfer in Speech Enhancement GAN (Pascual et al. 2018)](https://arxiv.org/abs/1712.06340)
3. [Whispered-to-voiced Alaryngeal Speech Conversion with GANs (Pascual et al. 2018)](https://arxiv.org/abs/1808.10687)

### Cite

If this code is of any help to your research, please cite our work:

```
@article{pascual2017segan,
  title={SEGAN: Speech Enhancement Generative Adversarial Network},
  author={Pascual, Santiago and Bonafonte, Antonio and Serr{\`a}, Joan},
  journal={arXiv preprint arXiv:1703.09452},
  year={2017}
}
```
