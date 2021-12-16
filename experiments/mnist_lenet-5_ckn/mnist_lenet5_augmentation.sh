save_path="../../results/mnist_lenet-5_ckn_augmentation/"
num_iters=500
labeling_burnin=100
labeling_method="matrix balancing"
seeds=(1 2 3 4 5 6 7 8 9 10)
gpu=0

num_labeled=50
lr_sup_init=-7
lr_semisup=-6

augment=16
batch_size_unlabeled=64

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --augment $augment --batch_size_unlabeled $batch_size_unlabeled --gpu $gpu \
    --labeling_burnin $labeling_burnin --labeling_method "$labeling_method" --lr_semisup $lr_semisup \
    --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path --seed $seed
done


num_labeled=100
lr_sup_init=-5
lr_semisup=-5

augment=8
batch_size_unlabeled=128

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --augment $augment --batch_size_unlabeled $batch_size_unlabeled --gpu $gpu \
    --labeling_burnin $labeling_burnin --labeling_method "$labeling_method" --lr_semisup $lr_semisup \
    --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path --seed $seed
done


num_labeled=150
lr_sup_init=-3
lr_semisup=-5

augment=8
batch_size_unlabeled=128

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --augment $augment --batch_size_unlabeled $batch_size_unlabeled --gpu $gpu \
    --labeling_burnin $labeling_burnin --labeling_method "$labeling_method" --lr_semisup $lr_semisup \
    --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path --seed $seed
done


num_labeled=200
lr_sup_init=-4
lr_semisup=-4

augment=4
batch_size_unlabeled=256

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --augment $augment --batch_size_unlabeled $batch_size_unlabeled --gpu $gpu \
    --labeling_burnin $labeling_burnin --labeling_method "$labeling_method" --lr_semisup $lr_semisup \
    --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path --seed $seed
done


num_labeled=250
lr_sup_init=-5
lr_semisup=-7

augment=4
batch_size_unlabeled=256

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --augment $augment --batch_size_unlabeled $batch_size_unlabeled --gpu $gpu \
    --labeling_burnin $labeling_burnin --labeling_method "$labeling_method" --lr_semisup $lr_semisup \
    --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path --seed $seed
done


num_labeled=300
lr_sup_init=-5
lr_semisup=-7

augment=4
batch_size_unlabeled=256

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --augment $augment --batch_size_unlabeled $batch_size_unlabeled --gpu $gpu \
    --labeling_burnin $labeling_burnin --labeling_method "$labeling_method" --lr_semisup $lr_semisup \
    --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path --seed $seed
done


num_labeled=350
lr_sup_init=-4
lr_semisup=-6

augment=2
batch_size_unlabeled=512

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --augment $augment --batch_size_unlabeled $batch_size_unlabeled --gpu $gpu \
    --labeling_burnin $labeling_burnin --labeling_method "$labeling_method" --lr_semisup $lr_semisup \
    --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path --seed $seed
done


num_labeled=400
lr_sup_init=-4
lr_semisup=-8

augment=4
batch_size_unlabeled=256

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --augment $augment --batch_size_unlabeled $batch_size_unlabeled --gpu $gpu \
    --labeling_burnin $labeling_burnin --labeling_method "$labeling_method" --lr_semisup $lr_semisup \
    --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path --seed $seed
done


num_labeled=450
lr_sup_init=-4
lr_semisup=-10

augment=2
batch_size_unlabeled=512

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --augment $augment --batch_size_unlabeled $batch_size_unlabeled --gpu $gpu \
    --labeling_burnin $labeling_burnin --labeling_method "$labeling_method" --lr_semisup $lr_semisup \
    --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path --seed $seed
done


num_labeled=500
lr_sup_init=-5
lr_semisup=-9

augment=2
batch_size_unlabeled=512

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --augment $augment --batch_size_unlabeled $batch_size_unlabeled --gpu $gpu \
    --labeling_burnin $labeling_burnin --labeling_method "$labeling_method" --lr_semisup $lr_semisup \
    --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path --seed $seed
done


num_labeled=0
labeling_burnin=0
lr_semisup=-2
lr_sup_init=-2
lam=-40

augment=64
batch_size_unlabeled=16

for seed in "${seeds[@]}"
do
  python mnist_lenet-5_ckn.py --augment $augment --batch_size_unlabeled $batch_size_unlabeled --eval_test_every 1 \
  --gpu $gpu --labeling_burnin $labeling_burnin --labeling_method "$labeling_method" --lam $lam \
  --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled \
  --save_path $save_path --seed $seed
done
