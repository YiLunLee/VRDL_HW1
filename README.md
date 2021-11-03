# VRDL_HW1: Bird Image Classification

## Installation
Build the enviroment via:
$ conda create -f requirements.yml

## Prepare Dataset


## Training Code
1. I use the Vision Transformer as my backbone model, and the official pre-trained (on ImageNet21K) weight can be downloaded in https://github.com/jeonsworld/ViT-pytorch In my experiments, I use the weight of ViT-B_16.npz trained on ImageNet21K.
2. Train the model on the given Bird Dataset. I run our experiments on 2x1080Ti with bacth size of 4 for each card.
```
  $ python -m torch.distributed.launch --nproc_per_node 2 train.py --name HW1 --output HW1 --dataset CUB_HW --model_type ViT-B_16 --pretrained_dir ./ViT-B_16.npz --img_size 448 --resize_size 600 --train_batch_size 8 --learning_rate 0.02 --num_steps 10000 --fp16 --eval_every 200 --feature_fusion
```

## Evaluation code
Evaluate the model on the test data.
```
```

## Download Pretrained Models
Here is the model weight of my final submission. Please download the weight and run the above evaluation code.
+ [Final_submission_weights]
## Reference
My howework references the codes in the following repos. Thanks for thier works and sharing.
+ [ViT-Pytorch](https://github.com/jeonsworld/ViT-pytorch)
+ [FFVT](https://github.com/Markin-Wang/FFVT#prerequisites)

