# Speech Enhancement Generative Adversarial Network in PyTorch


Execute training with default parameters giving latest best results in denoising and dewhispering:

```
python train.py --save_path ckpt_segan+ --batch_size 300 \
		--clean_trainset data/clean_trainset \
		--noisy_trainset data/noisy_trainset \
		--cache_dir data/cache
```

Read `run_segan+_train.sh` for more guidance

Clean files by specifying the generator weights checkpoint, its config file from training and appropriate paths for input and output files (Use `soundfile` wav writer backend (recommended) specifying the `--soundfile` flag):

```
python clean.py --g_pretrained_ckpt ckpt_segan+/<weights_ckpt_for_G> \
		--cfg_file ckpt_segan+/train.opts --synthesis_path enhanced_results \
		--test_files data/noisy_testset --soundfile
```

Read `run_segan+_clean.sh` for more guidance

### Disclaimer:

* Multi-GPU is not supported yet in this framework.
* Virtual Batch Norm is not included, and similar results to those of original paper can be obtained with regular BatchNorm in D (ONLY D).


### References:

* [SEGAN: Speech Enhancement Generative Adversarial Network (Pascual et al. 2017)](https://arxiv.org/abs/1703.09452)
* [Language and Noise Transfer in Speech Enhancement GAN (Pascual et al. 2018)](https://arxiv.org/abs/1712.06340)
* [Whispered-to-voiced Alaryngeal Speech Conversion with GANs (Pascual et al. 2018)](https://arxiv.org/abs/1808.10687)
