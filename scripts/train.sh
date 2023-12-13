#!/bin/bash
python train.py --train_clips_per_video 6 --val_clips_per_video 1 --data_dir ../dataSets/train_syn_data --ngpu 2 --nepoch 300 --nworker 8 --label_type beat_random_time_c_shift_scaled --lr 1e-6 --num_accS_dir 4 --num_velS_dir 4 --num_blurrs 4 --clip_size 72 --data_augmentation ../dataAugParams/data_augmentation_list.json --batch_size 16
