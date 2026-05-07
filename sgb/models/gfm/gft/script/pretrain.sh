#!/bin/bash

#$ -N Pretrain
#$ -pe smp 8
#$ -l gpu=1

~/.conda/envs/GFT/bin/python GFT/pretrain.py --use_params