#!/bin/bash

mkdir -p logs
# GPU 0: NODE, HNODE+TF10, FNODE+TF10
# GPU 1: LNODE+TF10, SNODE+TF10, RNODE+TF10

# 1) NODE baseline (GPU 0)
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python train.py \
  --system hindmarsh_rose --model node --epochs 1000 --seq_len 200 \
  --hidden 64 \
  --tvalid_every 20 --log_every 10 \
  --run_name hr_node &
echo "PID NODE: $!"

# 2) HNODE+TF10 (GPU 0)
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python train.py \
  --system hindmarsh_rose --model anode --epochs 1000 --seq_len 200 \
  --hidden 64 --aug 4 \
  --tvalid_every 20 --log_every 10 \
  --run_name hr_anode &
echo "PID HNODE: $!"

CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python train.py \
  --system hindmarsh_rose --model csode --epochs 1000 --seq_len 200 \
  --hidden 64 \
  --tvalid_every 20 --log_every 10 \
  --run_name hr_csode &
echo "PID SNODE: $!"

# 3) FNODE+TF10 (GPU 0)
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python train.py \
  --system hindmarsh_rose --model fnode --epochs 1000 --seq_len 200 \
  --hidden 64 --aug 4 --curriculum --curriculum_schedule 20:50,40:100,60:150,100:200,150:500 \
  --tvalid_every 20 --log_every 10 --tf_chunk 10 \
  --run_name hr_fnode_tf10 &
echo "PID FNODE: $!"

# 3) FNODE+TF10 (GPU 0)
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python train.py \
  --system hindmarsh_rose --model fnode_v2 --epochs 1000 --seq_len 200 \
  --hidden 64 --aug 4 --curriculum --curriculum_schedule 20:50,40:100,60:150,100:200,150:500 \
  --tvalid_every 20 --log_every 10 --tf_chunk 10 \
  --run_name hr_fnodev2_tf10 &
echo "PID FNODE: $!"

CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python train.py \
  --system hindmarsh_rose --model fnode_v2 --epochs 1000 --seq_len 500 \
  --hidden 96 --aug 4 --curriculum --curriculum_schedule 20:20,50:50,85:100,150:270,250:300,500:260   \
  --tvalid_every 20 --log_every 10 --tf_chunk 10 \
  --run_name hr_fnodev2_v5_tf10 &

CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python train.py \
  --system hindmarsh_rose --model fnode_v2 --epochs 500 --seq_len 500 \
  --hidden 96 --aug 0\
  --tvalid_every 20 --log_every 10 \
  --run_name hr_fnodev2_no_tf_cur &
