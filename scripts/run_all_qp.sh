#!/bin/bash

mkdir -p logs
# GPU 0: NODE, HNODE+TF10, FNODE+TF10
# GPU 1: LNODE+TF10, SNODE+TF10, RNODE+TF10

# 1) NODE baseline (GPU 0)
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python train.py \
  --system quasi_periodic --model node --epochs 1000 --seq_len 200 \
  --hidden 64 \
  --tvalid_every 20 --log_every 10 \
  --run_name qp_node &
echo "PID NODE: $!"

# 2) HNODE+TF10 (GPU 0)
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python train.py \
  --system quasi_periodic --model anode --epochs 1000 --seq_len 200 \
  --hidden 64 --aug 4 \
  --tvalid_every 20 --log_every 10 \
  --run_name qp_anode &
echo "PID HNODE: $!"

CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python train.py \
  --system quasi_periodic --model csode --epochs 1000 --seq_len 200 \
  --hidden 64 \
  --tvalid_every 20 --log_every 10 \
  --run_name qp_csode &
echo "PID SNODE: $!"

# 3) FNODE+TF10 (GPU 0)
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python train.py \
  --system quasi_periodic --model fnode --epochs 1000 --seq_len 200 \
  --hidden 64 --aug 4 --curriculum --curriculum_schedule 20:50,40:100,60:150,100:200,150:500 \
  --tvalid_every 20 --log_every 10 --tf_chunk 10 \
  --run_name qp_fnode_tf10 &
echo "PID FNODE: $!"

# 3) FNODE+TF10 (GPU 0)
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python train.py \
  --system quasi_periodic --model fnode_v2 --epochs 1000 --seq_len 200 \
  --hidden 64 --aug 4 --curriculum --curriculum_schedule 20:50,40:100,60:150,100:200,150:500 \
  --tvalid_every 20 --log_every 10 --tf_chunk 10 \
  --run_name qp_fnodev2_tf10 &
echo "PID FNODE: $!"

CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python train.py \
  --system quasi_periodic --model fnode_v2 --epochs 1000 --seq_len 500 \
  --hidden 96 --aug 4 --curriculum --curriculum_schedule 20:20,50:50,85:100,150:270,250:300,500:260   \
  --tvalid_every 20 --log_every 10 --tf_chunk 10 \
  --run_name qp_fnodev2_tf10 &

CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python train.py \
  --system quasi_periodic --model fnode_v2 --epochs 1000 --seq_len 500 \
  --hidden 96 --aug 4 --curriculum --curriculum_schedule 20:20,50:50,85:100,150:270,250:300,500:260   \
  --tvalid_every 20 --log_every 10 --tf_chunk 50 \
  --run_name qp_fnodev2_tf50 &


CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python train.py \
  --system quasi_periodic --model csode --epochs 1000 --seq_len 500 \
  --hidden 64 --curriculum --curriculum_schedule 20:20,50:50,85:100,150:270,250:300,500:260 \
  --tvalid_every 20 --log_every 10 --tf_chunk 50 \
  --run_name qp_csode_tf_cur &