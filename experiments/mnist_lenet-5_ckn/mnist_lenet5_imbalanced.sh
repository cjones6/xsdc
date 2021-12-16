save_path="../../results/mnist_lenet-5_ckn_imbalanced/"
labeling_burnin=100
labeling_method="matrix balancing"
lr_sup_init=-7
num_iters=500
num_labeled=50
seeds=(1 2 3 4 5 6 7 8 9 10)
gpu=0


imbalance=0.05
max_fracs=0.11
lr_semisup=-3

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --gpu $gpu --imbalance $imbalance --labeling_burnin $labeling_burnin \
    --labeling_method "$labeling_method" --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init \
    --max_frac_points_class $max_fracs --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path \
    --seed $seed
done


imbalance=0.1
max_fracs=0.1
lr_semisup=-5

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --gpu $gpu --imbalance $imbalance --labeling_burnin $labeling_burnin \
    --labeling_method "$labeling_method" --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init \
    --max_frac_points_class $max_fracs --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path \
    --seed $seed
done


imbalance=0.15
max_fracs=0.1
lr_semisup=-3

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --gpu $gpu --imbalance $imbalance --labeling_burnin $labeling_burnin \
    --labeling_method "$labeling_method" --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init \
    --max_frac_points_class $max_fracs --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path \
    --seed $seed
done


imbalance=0.2
max_fracs=0.14
lr_semisup=-5

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --gpu $gpu --imbalance $imbalance --labeling_burnin $labeling_burnin \
    --labeling_method "$labeling_method" --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init \
    --max_frac_points_class $max_fracs --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path \
    --seed $seed
done


imbalance=0.25
max_fracs=0.19
lr_semisup=-3

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --gpu $gpu --imbalance $imbalance --labeling_burnin $labeling_burnin \
    --labeling_method "$labeling_method" --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init \
    --max_frac_points_class $max_fracs --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path \
    --seed $seed
done


imbalance=0.30
max_fracs=0.1
lr_semisup=-3

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --gpu $gpu --imbalance $imbalance --labeling_burnin $labeling_burnin \
    --labeling_method "$labeling_method" --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init \
    --max_frac_points_class $max_fracs --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path \
    --seed $seed
done


imbalance=0.35
max_fracs=0.16
lr_semisup=-6

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --gpu $gpu --imbalance $imbalance --labeling_burnin $labeling_burnin \
    --labeling_method "$labeling_method" --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init \
    --max_frac_points_class $max_fracs --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path \
    --seed $seed
done


imbalance=0.4
max_fracs=0.1
lr_semisup=-5

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --gpu $gpu --imbalance $imbalance --labeling_burnin $labeling_burnin \
    --labeling_method "$labeling_method" --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init \
    --max_frac_points_class $max_fracs --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path \
    --seed $seed
done


imbalance=0.45
max_fracs=0.15
lr_semisup=-7

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --gpu $gpu --imbalance $imbalance --labeling_burnin $labeling_burnin \
    --labeling_method "$labeling_method" --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init \
    --max_frac_points_class $max_fracs --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path \
    --seed $seed
done


imbalance=0.5
max_fracs=0.1
lr_semisup=-4

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --gpu $gpu --imbalance $imbalance --labeling_burnin $labeling_burnin \
    --labeling_method "$labeling_method" --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init \
    --max_frac_points_class $max_fracs --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path \
    --seed $seed
done


imbalance=0.55
max_fracs=0.19
lr_semisup=-7

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --gpu $gpu --imbalance $imbalance --labeling_burnin $labeling_burnin \
    --labeling_method "$labeling_method" --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init \
    --max_frac_points_class $max_fracs --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path \
    --seed $seed
done


imbalance=0.6
max_fracs=0.2
lr_semisup=-4

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --gpu $gpu --imbalance $imbalance --labeling_burnin $labeling_burnin \
    --labeling_method "$labeling_method" --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init \
    --max_frac_points_class $max_fracs --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path \
    --seed $seed
done


imbalance=0.65
max_fracs=0.11
lr_semisup=-5

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --gpu $gpu --imbalance $imbalance --labeling_burnin $labeling_burnin \
    --labeling_method "$labeling_method" --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init \
    --max_frac_points_class $max_fracs --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path \
    --seed $seed
done


imbalance=0.7
max_fracs=0.1
lr_semisup=-5

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --gpu $gpu --imbalance $imbalance --labeling_burnin $labeling_burnin \
    --labeling_method "$labeling_method" --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init \
    --max_frac_points_class $max_fracs --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path \
    --seed $seed
done


imbalance=0.75
max_fracs=0.15
lr_semisup=-4

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --gpu $gpu --imbalance $imbalance --labeling_burnin $labeling_burnin \
    --labeling_method "$labeling_method" --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init \
    --max_frac_points_class $max_fracs --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path \
    --seed $seed
done


imbalance=0.8
max_fracs=0.1
lr_semisup=-6

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --gpu $gpu --imbalance $imbalance --labeling_burnin $labeling_burnin \
    --labeling_method "$labeling_method" --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init \
    --max_frac_points_class $max_fracs --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path \
    --seed $seed
done


imbalance=0.85
max_fracs=0.1
lr_semisup=-9

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --gpu $gpu --imbalance $imbalance --labeling_burnin $labeling_burnin \
    --labeling_method "$labeling_method" --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init \
    --max_frac_points_class $max_fracs --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path \
    --seed $seed
done


imbalance=0.9
max_fracs=0.1
lr_semisup=-9

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --gpu $gpu --imbalance $imbalance --labeling_burnin $labeling_burnin \
    --labeling_method "$labeling_method" --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init \
    --max_frac_points_class $max_fracs --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path \
    --seed $seed
done


imbalance=0.95
max_fracs=0.1
lr_semisup=-1

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --gpu $gpu --imbalance $imbalance --labeling_burnin $labeling_burnin \
    --labeling_method "$labeling_method" --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init \
    --max_frac_points_class $max_fracs --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path \
    --seed $seed
done
