#!/bin/bash

#$ -N Finetune
#$ -pe smp 8
#$ -l gpu=1

for dataset in cora pubmed arxiv wikics WN18RR FB15K237 chemhiv chempcba
do
    ~/.conda/envs/GFT/bin/python GFT/finetune.py --use_params --dataset $dataset
done