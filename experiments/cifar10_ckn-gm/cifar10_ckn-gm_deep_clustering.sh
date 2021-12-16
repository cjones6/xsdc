save_path="../../results/cifar10_ckn-gm_deep_clustering/"
num_iters=500
labeling_burnin=100
labeling_method="deep clustering"
seeds=(1 2 3 4 5 6 7 8 9 10)
gpu=0


num_labeled=50
lr_sup_init=3
lr_semisup=-10
num_clusters=10
update_clusters_every=50

for seed in "${seeds[@]}"
do
    python cifar10_ckn-gm.py --gpu $gpu --labeling_burnin $labeling_burnin --labeling_method "$labeling_method" \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_clusters $num_clusters --num_iters $num_iters \
    --num_labeled $num_labeled --save_path $save_path --seed $seed --update_clusters_every $update_clusters_every
done


num_labeled=100
lr_sup_init=1
lr_semisup=-10
num_clusters=10
update_clusters_every=50

for seed in "${seeds[@]}"
do
    python cifar10_ckn-gm.py --gpu $gpu --labeling_burnin $labeling_burnin --labeling_method "$labeling_method" \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_clusters $num_clusters --num_iters $num_iters \
    --num_labeled $num_labeled --save_path $save_path --seed $seed --update_clusters_every $update_clusters_every
done


num_labeled=150
lr_sup_init=3
lr_semisup=3
num_clusters=10
update_clusters_every=50

for seed in "${seeds[@]}"
do
    python cifar10_ckn-gm.py --gpu $gpu --labeling_burnin $labeling_burnin --labeling_method "$labeling_method" \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_clusters $num_clusters --num_iters $num_iters \
    --num_labeled $num_labeled --save_path $save_path --seed $seed --update_clusters_every $update_clusters_every
done


num_labeled=200
lr_sup_init=3
lr_semisup=-10
num_clusters=10
update_clusters_every=50

for seed in "${seeds[@]}"
do
    python cifar10_ckn-gm.py --gpu $gpu --labeling_burnin $labeling_burnin --labeling_method "$labeling_method" \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_clusters $num_clusters --num_iters $num_iters \
    --num_labeled $num_labeled --save_path $save_path --seed $seed --update_clusters_every $update_clusters_every
done


num_labeled=250
lr_sup_init=4
lr_semisup=4
num_clusters=10
update_clusters_every=50

for seed in "${seeds[@]}"
do
    python cifar10_ckn-gm.py --gpu $gpu --labeling_burnin $labeling_burnin --labeling_method "$labeling_method" \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_clusters $num_clusters --num_iters $num_iters \
    --num_labeled $num_labeled --save_path $save_path --seed $seed --update_clusters_every $update_clusters_every
done


num_labeled=300
lr_sup_init=0
lr_semisup=5
num_clusters=10
update_clusters_every=50

for seed in "${seeds[@]}"
do
    python cifar10_ckn-gm.py --gpu $gpu --labeling_burnin $labeling_burnin --labeling_method "$labeling_method" \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_clusters $num_clusters --num_iters $num_iters \
    --num_labeled $num_labeled --save_path $save_path --seed $seed --update_clusters_every $update_clusters_every
done


num_labeled=350
lr_sup_init=3
lr_semisup=2
num_clusters=10
update_clusters_every=50

for seed in "${seeds[@]}"
do
    python cifar10_ckn-gm.py --gpu $gpu --labeling_burnin $labeling_burnin --labeling_method "$labeling_method" \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_clusters $num_clusters --num_iters $num_iters \
    --num_labeled $num_labeled --save_path $save_path --seed $seed --update_clusters_every $update_clusters_every
done


num_labeled=400
lr_sup_init=-1
lr_semisup=4
num_clusters=10
update_clusters_every=50

for seed in "${seeds[@]}"
do
    python cifar10_ckn-gm.py --gpu $gpu --labeling_burnin $labeling_burnin --labeling_method "$labeling_method" \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_clusters $num_clusters --num_iters $num_iters \
    --num_labeled $num_labeled --save_path $save_path --seed $seed --update_clusters_every $update_clusters_every
done


num_labeled=450
lr_sup_init=3
lr_semisup=-10
num_clusters=10
update_clusters_every=50

for seed in "${seeds[@]}"
do
    python cifar10_ckn-gm.py --gpu $gpu --labeling_burnin $labeling_burnin --labeling_method "$labeling_method" \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_clusters $num_clusters --num_iters $num_iters \
    --num_labeled $num_labeled --save_path $save_path --seed $seed --update_clusters_every $update_clusters_every
done


num_labeled=500
lr_sup_init=5
lr_semisup=4
num_clusters=10
update_clusters_every=50

for seed in "${seeds[@]}"
do
    python cifar10_ckn-gm.py --gpu $gpu --labeling_burnin $labeling_burnin --labeling_method "$labeling_method" \
    --lr_semisup $lr_semisup --lr_sup_init $lr_sup_init --num_clusters $num_clusters --num_iters $num_iters \
    --num_labeled $num_labeled --save_path $save_path --seed $seed --update_clusters_every $update_clusters_every
done


num_labeled=0
labeling_burnin=0
lr=5
lam=0
num_clusters=10
update_clusters_every=50

for seed in "${seeds[@]}"
do
    python cifar10_ckn-gm.py --eval_test_every 1 --gpu $gpu --labeling_burnin $labeling_burnin \
    --labeling_method "$labeling_method" --lam $lam --lr_semisup $lr --num_clusters $num_clusters \
    --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path --seed $seed \
    --update_clusters_every $update_clusters_every
done
