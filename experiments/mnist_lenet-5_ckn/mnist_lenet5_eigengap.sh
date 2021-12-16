save_path="../../results/mnist_lenet-5_ckn_eigendecomposition_kmeans/"
eval_test_every=1
gpu=0
labeling_burnin=0
labeling_method=eigendecomposition
lam=-10
lr=-4
min_frac=0.1
num_iters=50
num_labeled=0
rounding=k-means
seeds=(1 2 3 4 5 6 7 8 9 10)


for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --eval_test_every $eval_test_every --gpu $gpu  --labeling_burnin $labeling_burnin \
    --labeling_method $labeling_method --lam $lam --lr_semisup $lr --lr_sup_init $lr \
    --min_frac_points_class $min_frac --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path \
    --seed $seed --update_lambda 0 --rounding $rounding
done


save_path="../../results/mnist_lenet-5_ckn_eigendecomposition_no_rounding/"
lam=-11
lr=-5

for seed in "${seeds[@]}"
do
    python mnist_lenet-5_ckn.py --eval_test_every $eval_test_every --gpu $gpu  --labeling_burnin $labeling_burnin \
    --labeling_method $labeling_method --lam $lam --lr_semisup $lr --lr_sup_init $lr \
    --min_frac_points_class $min_frac --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path \
    --seed $seed --update_lambda 0
done
