#!/bin/bash

python train.py \
--trial 800 \
--model PROG_RNN_DET \
--x_dim 2 \
--y_dim 10 \
--z_dim 16 \
--h_dim 200 \
--m_dim 90 \
--rnn_dim 200 \
--rnn_micro_dim 400 \
--rnn_macro_dim 200 \
--rnn_mid_dim 300 \
--n_agents 5 \
--n_layers 2 \
--n_epochs 200 \
--clip 10 \
--start_lr 1e-3 \
--min_lr 1e-4 \
--batch_size 128 \
--pretrain 50 \
--burn_in 0 \
--subsample 2 \
--cuda \
--cont
