# Generative Latent Flow

A pytorch implementation of "Generative Latent Flow"


### Prerequisites

- `numpy`, `pytorch` and `tqdm`


### Training

- Default setting `python train.py` will train a model for cifar10.
- To train on mnist, use `python train.py --dataset mnist --num_latent 20 --image_size 28`
- To train on fashion mnist, use `python train.py --dataset Fashion-mnist --num_latent 20 --image_size 28`
- To train on celeba, use `python train.py --dataset celeba --num_latent 64 --image_size 64`, this one doesn't support automatically downloading dataset.
- Use `--loss_type` for different loss function, choices include `MSE` `Perceptual` `cross_entropy` (last one only for mnist and fashion mnist).

### Update 6/6/2019
Add `SigmoidCouplingLayer`, `--coupling x`, `x` for 0 (additive coupling), 1 (affine coupling) and 2 (sigmoid coupling).
Add class-based conditional model. `--num_class 10` for mnist, fmnist and cifar10.

### Todo: add learning rate decay scheme.

