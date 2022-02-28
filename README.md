# vision-transformers-cifar10
Let's train vision transformers for cifar 10! 

This is an unofficial and elementary implementation of `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`.

I use pytorch for implementation.


# Usage
`python train_cifar10.py --net vit_timm --lr 1e-4 --label-smoothing` # train with pretrained vit

# Results..

|             | Accuracy |
|:-----------:|:--------:|
| ViT small (timm transfer) | 97.5% |
| ViT base (timm transfer) | 98.5% |
