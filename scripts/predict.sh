#!/bin/bash
python predict.py --data_dir ../dataSets/bouncingBall --ngpu 1 --nepoch 1 --nworker 4 --label_type none --num_accS_dir 4 --num_velS_dir 4 --num_blurrs 4 --prediction_window_step 24 --load_model --load_model_dir ../checkpoints/caccvel_full --load_epoch 61
