#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
  -W ignore \
  main.py \
  --network_feature resnet18 \
  --dataset finegym \
  --batch_size 32 \
  --img_dim 128 \
  --hyperbolic \
  --hyperbolic_version 1 \
  --pred_step 0 \
  --seq_len 5 \
  --num_seq 12 \
  --distance 'squared' \
  --lr 1e-3 \
  --prefix test_future_subaction_linear_finegym_hyperbolic \
  --fp16 \
  --fp64_hyper \
  --pretrain /vulcan_data/roye_/hyperfuture/pretrained_models/checkpoints/train_finegym_hyperbolic/checkpoint.pth.tar \
  --linear_input predictions_z_hat \
  --n_classes 307 \
  --hierarchical_labels \
  --use_labels \
  --only_train_linear \
  --pred_future \
  --test \
  --num_workers 2 \
  --seed 0 \
  --path_dataset /vulcan_data/Finegym \
  --path_data_info /vulcan_data/Finegym/dataset_info 