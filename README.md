# VRDL_HW1: Bird Image Classification
This is homework 1 in NYCU Selected Topics in Visual Recognition using Deep Learning.

## Installation
**Requirements:**
+ CUDA version 10.1
+ pytorch = 1.5.1
+ torchvision = 0.6.1
+ ml-collection
+ scipy
+ tensorboard

Or build the environment via:
```
$ conda create -f environment.yml
```

## Prepare Dataset
Put the train image in the train folder: ./dataset/train
Put the test image in the test folder: ./dataset/test
The dataset folder should be like:
```
./dataset
  |---classes.txt
  |---train_set_labels.txt
  |---val_set_labels.txt
  |---training_labels.txt
  |---testing_img_order.txt
  |---train
        |---xxxx.png
        |---xxxx.png
              .
              .
              .
  |---test
        |---xxxx.png
        |---xxxx.png
              .
              .
              .
  
```

## Training Code
1. I follow the related work **FFVT** to use the Vision Transformer as my backbone model. The official pre-trained (on ImageNet21K) weight can be downloaded in https://github.com/jeonsworld/ViT-pytorch In my experiments, **I use the weight of ViT-B_16.npz trained on ImageNet21K**.
2. Train the model on the given Bird Dataset. I run our experiments on 2x1080Ti with bacth size of 4 for each card.
```
  $ python -m torch.distributed.launch --nproc_per_node 2 train.py --name [your exp name] --output_dir [your output dir] --dataset CUB_HW --model_type ViT-B_16 --pretrained_dir ./ViT-B_16.npz --img_size 448 --resize_size 600 --train_batch_size 4 --eval_batch_size 4 --learning_rate 0.02 --num_steps 10000 --eval_every 200 --feature_fusion
```

## Evaluation code
Evaluate the model on the test data.
```
  $ python inference.py --name Test --output_dir Submission_check --dataset CUB_HW --model_type ViT-B_16 --pretrained_dir [path to your pre-trained model] --img_size 448 --resize_size 600 --eval_batch_size 4 --feature_fusion
```

## Download Pretrained Models
Here is the model weight of my final submission. Please download the weights and run the above evaluation code.
+ [Final_submission_weights](https://drive.google.com/file/d/1-ZPSmGpu7uHxtfBe94kL_LEK5nIXjTwm/view?usp=sharing)

## Reference
My howework references the codes in the following repos. Thanks for thier works and sharing.
+ [ViT-Pytorch](https://github.com/jeonsworld/ViT-pytorch)
+ [FFVT](https://github.com/Markin-Wang/FFVT#prerequisites)

