save_path="../../results/mnist_lenet-5_ckn_mustnt_link_specific/"
constraints=specific
num_iters=500
labeling_burnin=100
seeds=(1 2 3 4 5 6 7 8 9 10)
gpu=0


num_labeled=50
lr_sup_init=-7
lr_semisup=-5

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --constraints $constraints --gpu $gpu --labeling_burnin $labeling_burnin \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled \
    --save_path $save_path --seed $seed
done


num_labeled=100
lr_sup_init=-5
lr_semisup=-7

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --constraints $constraints --gpu $gpu --labeling_burnin $labeling_burnin \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled \
    --save_path $save_path --seed $seed
done


num_labeled=150
lr_sup_init=-3
lr_semisup=-3

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --constraints $constraints --gpu $gpu --labeling_burnin $labeling_burnin \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled \
    --save_path $save_path --seed $seed
done


num_labeled=200
lr_sup_init=-4
lr_semisup=-5

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --constraints $constraints --gpu $gpu --labeling_burnin $labeling_burnin \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled \
    --save_path $save_path --seed $seed
done


num_labeled=250
lr_sup_init=-5
lr_semisup=-6

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --constraints $constraints --gpu $gpu --labeling_burnin $labeling_burnin \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled \
    --save_path $save_path --seed $seed
done


num_labeled=300
lr_sup_init=-5
lr_semisup=-7

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --constraints $constraints --gpu $gpu --labeling_burnin $labeling_burnin \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled \
    --save_path $save_path --seed $seed
done


num_labeled=350
lr_sup_init=-4
lr_semisup=-8

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --constraints $constraints --gpu $gpu --labeling_burnin $labeling_burnin \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled \
    --save_path $save_path --seed $seed
done


num_labeled=400
lr_sup_init=-4
lr_semisup=-5

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --constraints $constraints --gpu $gpu --labeling_burnin $labeling_burnin \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled \
    --save_path $save_path --seed $seed
done


num_labeled=450
lr_sup_init=-4
lr_semisup=-8

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --constraints $constraints --gpu $gpu --labeling_burnin $labeling_burnin \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled \
    --save_path $save_path --seed $seed
done


num_labeled=500
lr_sup_init=-5
lr_semisup=-9

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --constraints $constraints --gpu $gpu --labeling_burnin $labeling_burnin \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled \
    --save_path $save_path --seed $seed
done


save_path="../../results/mnist_lenet-5_ckn_mustnt_link_random/"
constraints=random


num_labeled=50
lr_sup_init=-7
lr_semisup=-3

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --constraints $constraints --gpu $gpu --labeling_burnin $labeling_burnin \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled \
    --save_path $save_path --seed $seed
done

num_labeled=100
lr_sup_init=-5
lr_semisup=-5

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --constraints $constraints --gpu $gpu --labeling_burnin $labeling_burnin \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled \
    --save_path $save_path --seed $seed
done


num_labeled=150
lr_sup_init=-3
lr_semisup=-3

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --constraints $constraints --gpu $gpu --labeling_burnin $labeling_burnin \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled \
    --save_path $save_path --seed $seed
done


num_labeled=200
lr_sup_init=-4
lr_semisup=-4

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --constraints $constraints --gpu $gpu --labeling_burnin $labeling_burnin \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled \
    --save_path $save_path --seed $seed
done

num_labeled=250
lr_sup_init=-5
lr_semisup=-6

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --constraints $constraints --gpu $gpu --labeling_burnin $labeling_burnin \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled \
    --save_path $save_path --seed $seed
done

num_labeled=300
lr_sup_init=-5
lr_semisup=-7

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --constraints $constraints --gpu $gpu --labeling_burnin $labeling_burnin \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled \
    --save_path $save_path --seed $seed
done

num_labeled=350
lr_sup_init=-4
lr_semisup=-8

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --constraints $constraints --gpu $gpu --labeling_burnin $labeling_burnin \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled \
    --save_path $save_path --seed $seed
done

num_labeled=400
lr_sup_init=-4
lr_semisup=-4

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --constraints $constraints --gpu $gpu --labeling_burnin $labeling_burnin \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled \
    --save_path $save_path --seed $seed
done

num_labeled=450
lr_sup_init=-4
lr_semisup=-9

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --constraints $constraints --gpu $gpu --labeling_burnin $labeling_burnin \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled \
    --save_path $save_path --seed $seed
done

num_labeled=500
lr_sup_init=-5
lr_semisup=-7

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --constraints $constraints --gpu $gpu --labeling_burnin $labeling_burnin \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled \
    --save_path $save_path --seed $seed
done
