#!/usr/bin/env bash

GPU=$1
MODEL=$2
# homo; hetero
DISTRIBUTION=$3
ROUND=$4
EPOCH=$5
BATCH_SIZE=$6

hostname > mpi_host_file

mpirun -np 4 -hostfile ./mpi_host_file python3 ./main.py \
  --gpu $GPU \
  --model $MODEL \
  --dataset cifar10 \
  --partition $DISTRIBUTION  \
  --client_number 4 \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --batch_size $BATCH_SIZE