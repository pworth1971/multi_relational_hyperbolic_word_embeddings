CUDA_VISIBLE_DEVICES=0 python ./multi_relational_training.py --model poincare --dataset cpae --num_iterations 4 --nneg 50 --batch_size 128 --lr 50 --dim 300  --device mps > output/run.mps.out
