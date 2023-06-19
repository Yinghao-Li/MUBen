#!/bin/bash

# Quit if there are any errors
set -e

# --- toggle the following command to decide whether train the model on each dataset ---
# --- the argument values do not matter ---

# -- regression tasks --
train_on_esol=true
train_on_freesolv=true
train_on_lipo=true
train_on_qm7=true
train_on_qm8=true
train_on_qm9=true

# -- single-task classification --
train_on_bbbp=true
train_on_bace=true
train_on_hiv=true

# -- multi-task classification --
train_on_tox21=true
train_on_toxcast=true
train_on_clintox=true
train_on_sider=true
train_on_muv=true

# --- dataset choosing region ends ---

# --- universal arguments ---
cuda_device=$1

disable_wandb=false

data_folder="./data/files"
num_workers=0
num_preprocess_workers=24
pin_memory=false
ignore_preprocessed_dataset=false

uncertainty_method="none"  # this is subject to change
retrain_model=false

regression_with_variance=true

lr=0.00005
batch_size=128
n_epochs=200
valid_tolerance=40

# Uncertainty arguments
n_test=30
n_ensembles=10

n_ts_epochs=30

# --- universal arguments region ends ---

# construct the list of datasets used for training
dataset_names=""
if [ -z ${train_on_bbbp+x} ]; then echo "skip bbbp"; else dataset_names+=" bbbp"; fi
if [ -z ${train_on_bace+x} ]; then echo "skip bace"; else dataset_names+=" bace"; fi
if [ -z ${train_on_tox21+x} ]; then echo "skip tox21"; else dataset_names+=" tox21"; fi
if [ -z ${train_on_toxcast+x} ]; then echo "skip toxcast"; else dataset_names+=" toxcast"; fi
if [ -z ${train_on_clintox+x} ]; then echo "skip clintox"; else dataset_names+=" clintox"; fi
if [ -z ${train_on_sider+x} ]; then echo "skip sider"; else dataset_names+=" sider"; fi
if [ -z ${train_on_esol+x} ]; then echo "skip esol"; else dataset_names+=" esol"; fi
if [ -z ${train_on_freesolv+x} ]; then echo "skip freesolv"; else dataset_names+=" freesolv"; fi
if [ -z ${train_on_lipo+x} ]; then echo "skip lipo"; else dataset_names+=" lipo"; fi
if [ -z ${train_on_qm7+x} ]; then echo "skip qm7"; else dataset_names+=" qm7"; fi
if [ -z ${train_on_qm8+x} ]; then echo "skip qm8"; else dataset_names+=" qm8"; fi
# large datasets have lower priority
if [ -z ${train_on_hiv+x} ]; then echo "skip hiv"; else dataset_names+=" hiv"; fi
if [ -z ${train_on_muv+x} ]; then echo "skip muv"; else dataset_names+=" muv"; fi
if [ -z ${train_on_qm9+x} ]; then echo "skip qm9"; else dataset_names+=" qm9"; fi
if [ -z ${train_on_pcba+x} ]; then echo "skip pcba"; else dataset_names+=" pcba"; fi

# --- training scripts ---
for dataset_name in $dataset_names
do
  for seed in 0 1 2
  do
    CUDA_VISIBLE_DEVICES=$cuda_device \
    PYTHONPATH="." \
    python ./run/chemberta.py \
      --disable_wandb $disable_wandb \
      --data_folder $data_folder \
      --dataset_name "$dataset_name" \
      --num_workers $num_workers \
      --num_preprocess_workers $num_preprocess_workers \
      --pin_memory $pin_memory \
      --ignore_preprocessed_dataset $ignore_preprocessed_dataset \
      --uncertainty_method $uncertainty_method \
      --retrain_model $retrain_model \
      --regression_with_variance $regression_with_variance \
      --lr $lr \
      --n_epochs $n_epochs \
      --valid_tolerance $valid_tolerance \
      --batch_size $batch_size \
      --seed $seed \
      --n_test $n_test \
      --n_ensembles $n_ensembles \
      --n_ts_epochs $n_ts_epochs \
      --deploy
  done
done
