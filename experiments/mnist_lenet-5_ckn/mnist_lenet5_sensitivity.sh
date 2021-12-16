save_path="../../results/mnist_lenet-5_ckn_sensitivity_analysis_labeled_lr/"
num_iters=500
num_labeled=50
labeling_burnin=100
labeling_method="matrix balancing"
seeds=(1 2 3 4 5 6 7 8 9 10)
gpu=0


num_labeled=50
lr_sup_inits=(-10 -9 -8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5)
lr_semisup=-6

for seed in "${seeds[@]}"
do
    for lr_sup_init in "${lr_sup_inits[@]}"
    do
      python mnist_lenet-5_ckn.py --gpu $gpu --labeling_burnin $labeling_burnin --labeling_method "$labeling_method" \
      --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled \
      --save_path $save_path --seed $seed
    done
done


save_path="../../results/mnist_lenet-5_ckn_sensitivity_analysis_labeled_unlabeled_lr/"
lr_semisups=(-10 -9 -8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5)
lr_sup_init=-7

for seed in "${seeds[@]}"
do
    for lr_semisup in "${lr_semisups[@]}"
    do
      python mnist_lenet-5_ckn.py --gpu $gpu --labeling_burnin $labeling_burnin --labeling_method "$labeling_method" \
      --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled \
      --save_path $save_path --seed $seed
    done
done


save_path="../../results/mnist_lenet-5_ckn_sensitivity_analysis_lambda/"
lr_sup_init=-7
lr_semisup=-6
lambdas=(-40 -35 -30 -25 -20 -15 -10 -5 0)
update_lambda=0

for seed in "${seeds[@]}"
do
    for lambda in "${lambdas[@]}"
    do
        python mnist_lenet-5_ckn.py --gpu $gpu --labeling_burnin $labeling_burnin --lam $lambda \
        --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_iters $num_iters --num_labeled $num_labeled \
        --save_path $save_path --seed $seed --update_lambda $update_lambda
    done
done
