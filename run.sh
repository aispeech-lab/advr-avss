# /bin/bash

CUDA_VISIBLE_DEVICES=2 nohup python3 -u train.py --gpus=1 \
    --TensorboardPath=./log/tb_test \
    --ParamsPath=./params/params_test \
    --seed=1 \
    --e_id=1 \
    >./log/test.log 2>&1 &

tail -f ./log/test.log