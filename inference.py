# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta
import time

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    if args.feature_fusion:
        config.feature_fusion = True
    config.num_token = args.num_token

    if args.dataset == "CUB_HW":
        num_classes = 200

    model = VisionTransformer(config, args.img_size, zero_head=True,
                              num_classes=num_classes, vis=True,
                              smoothing_value=args.smoothing_value,
                              dataset=args.dataset)
#     model.load_from(np.load(args.pretrained_dir, allow_pickle=True))
    model.load_state_dict(torch.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def inference(args, model):
    """ Test the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    best_step = 0

    args.train_batch_size = (args.train_batch_size
                             // args.gradient_accumulation_steps)

    # Prepare dataset
    train_loader, valid_loader, test_loader = get_loader(args)

    t_total = args.num_steps

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])])

    # Testing!
    eval_losses = AverageMeter()

    logger.info("***** Running Test data *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    all_preds, all_names = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Testing... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        x, name = batch
        x = x.to(args.device)
        with torch.no_grad():
            logits = model(x)[0]

            preds = torch.argmax(logits, dim=-1)
        for idx in range(x.size(0)):
            all_preds.append(preds[idx].item())
            all_names.append(name[idx])

        epoch_iterator.set_description("Testing... (loss=%2.5f)"
                                       % eval_losses.val)

    with open('./dataset/classes.txt', 'r') as f:
        label_list = [x.strip() for x in f.readlines()]
    with open(os.path.join(args.output_dir, 'answer.txt'), 'w') as f:
        for idx in range(len(all_preds)):
            pred_label = all_preds[idx]
            f.write('{} {}\n'.format(all_names[idx], label_list[pred_label]))


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "CUB_HW"],
                        default="cotton",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32",
                                                 "ViT-L_16", "ViT-L_32",
                                                 "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str,
                        default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir",
                        default="output", type=str,
                        help="The output directory for checkpoints.")

    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument("--resize_size", default=600, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--num_token", default=12, type=int,
                        help="the number of selected token in each layer.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every N steps.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"],
                        default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Steps to perform learning rate warmup.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate "
                             "before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision")
    parser.add_argument('--feature_fusion', action='store_true',
                        help="Whether to use feature fusion")
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")

    args = parser.parse_args()
    args.data_root = '{}/{}'.format(args.data_root, args.dataset)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which
        # will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device
    args.nprocs = torch.cuda.device_count()

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, \
                    distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device,
                    args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # inference
    inference(args, model)


if __name__ == "__main__":
    main()
