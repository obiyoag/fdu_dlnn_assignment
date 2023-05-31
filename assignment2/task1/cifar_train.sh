CUDA_VISIBLE_DEVICES=0 python train_baseline.py &
CUDA_VISIBLE_DEVICES=0 python train_cutout.py &
CUDA_VISIBLE_DEVICES=0 python train_cutmix.py &
CUDA_VISIBLE_DEVICES=0 python train_mixup.py &