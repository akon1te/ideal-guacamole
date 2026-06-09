#!/bin/bash

mkdir -p logs

# 1) NODE baseline (GPU 0)
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python train.py \
  --system rossler --model node --epochs 1000 --seq_len 300 \
  --hidden 64 \
  --tvalid_every 20 --log_every 10 \
  --run_name rossler_node &
echo "PID NODE: $!"

# 2) ANODE (GPU 0)
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python train.py \
  --system rossler --model anode --epochs 1000 --seq_len 300 \
  --hidden 64 --aug 4 \
  --tvalid_every 20 --log_every 10 \
  --run_name rossler_anode &
echo "PID HNODE: $!"

# 5) CSODE (GPU 1)
CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python train.py \
  --system rossler --model csode --epochs 1000 --seq_len 300 \
  --hidden 64 \
  --tvalid_every 20 --log_every 10 \
  --run_name rossler_csode &
echo "PID SNODE: $!"

# 3) FNODE+TF10 (GPU 0)
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python train.py \
  --system rossler --model fnode --epochs 1000 --seq_len 300 \
  --hidden 64 --aug 4 --curriculum --curriculum_schedule 20:20,50:50,85:100,150:270,250:500 \
  --tvalid_every 20 --log_every 10 --tf_chunk 10 \
  --run_name rossler_fnode_tf10 &
echo "PID FNODE: $!"

# 3) FNODE_V2+TF10 (GPU 0)
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python train.py \
  --system rossler --model fnode_v2 --epochs 1000 --seq_len 300 \
  --hidden 64 --aug 4 --curriculum --curriculum_schedule 20:20,50:50,85:100,150:270,250:500  \
  --tvalid_every 20 --log_every 10 --tf_chunk 10 \
  --run_name rossler_fnodev2_tf10 &
echo "PID FNODE: $!"

CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python train.py \
  --system rossler --model fnode_v2 --epochs 1000 --seq_len 300 \
  --hidden 128 --aug 4 --curriculum --curriculum_schedule 20:20,50:50,85:100,150:270,250:500  \
  --tvalid_every 20 --log_every 10 --tf_chunk 10 \
  --run_name rossler_fnodev3_tf10 &
echo "PID FNODE: $!"

CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python train.py \
  --system rossler --model fnode_v2 --epochs 1000 --seq_len 500 \
  --hidden 128 --aug 4 --curriculum --curriculum_schedule 20:10,50:30,100:60,150:80,250:100,400:150,500:70  \
  --tvalid_every 20 --log_every 10 --tf_chunk 10 --tvalid_threshold 0.5 \
  --run_name rossler_fnodev4_tf10 &
echo "PID FNODE: $!"

CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python train.py \
  --system rossler --model fnode_v2 --epochs 1000 --seq_len 500 \
  --hidden 96 --aug 4 --curriculum --curriculum_schedule 20:20,50:50,85:100,150:270,250:300,500:260   \
  --tvalid_every 20 --log_every 10 --tf_chunk 10 \
  --run_name rossler_fnodev2_v5_tf10 &
echo "PID FNODE: $!"

