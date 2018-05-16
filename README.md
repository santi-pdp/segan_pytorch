# Speech Enhancement Generative Adversarial Network in PyTorch

Train SEGAN replica:


Execute training with default parameters (kwidth=31, skip connections with concat merge, etc.), CUDA, a `batch_size` of 100:

```
python train.py --save_path seganv1_ckpt --cuda --batch_size 100 \
		--clean_trainset data/clean_trainset \
		--noisy_trainset data/noisy_trainset \
		--clean_valset data/clean_valset \
		--noisy_valset data/noisy_valset \
		--cache_dir data/cache
```

### TODO: Write clean script


### Disclaimer:

* Multi-GPU is not supported yet in this framework.
* Virtual Batch Norm is not included, and similar results to those of original paper can be obtained with regular BatchNorm in D (ONLY D).

