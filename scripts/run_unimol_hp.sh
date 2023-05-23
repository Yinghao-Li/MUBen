#!/bin/bash

# Quit if there are any errors
set -e

# --- toggle the following command to decide whether train the model on each dataset ---
# --- the argument values do not matter ---

# -- regression tasks --
#train_on_esol=true  # fine-tuned
#train_on_freesolv=true  # fine-tuned
#train_on_lipo=true  # fine-tuned
#train_on_qm7=true  # fine-tuned
#train_on_qm8=true  # fine-tuned
#train_on_qm9=true  # fine-tuned

# -- single-task classification --
# train_on_bbbp=true  # fine-tuned
# train_on_bace=true  # fine-tuned
#train_on_hiv=true

# -- multi-task classification --
#train_on_tox21=true  # fine-tuned
#train_on_toxcast=true  # fine-tuned
#train_on_clintox=true
train_on_sider=true
#train_on_muv=true
# train_on_pcba=true

# --- dataset choosing region ends ---

# --- universal arguments ---
cuda_device=$1

wandb_api_key="86efcc8aa38a82bd8128db4c1cee3acde0f33920"
disable_wandb=false

data_folder="./data/files"
unimol_feature_folder="./data/UniMol"
num_workers=16
num_preprocess_workers=16
pin_memory=false
ignore_preprocessed_dataset=true

uncertainty_method="none"  # this is subject to change
retrain_model=true

binary_classification_with_softmax=false
regression_with_variance=true

lr=0.0005
batch_size=32
batch_size_inference=16
n_epochs=80
valid_tolerance=40

seed=0

# Uncertainty arguments
n_test=1

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
if [ -z ${train_on_pcab+x} ]; then echo "skip pcab"; else dataset_names+=" pcab"; fi

# --- training scripts ---
for dataset_name in $dataset_names
do
  CUDA_VISIBLE_DEVICES=$cuda_device python run_unimol.py \
    --wandb_api_key $wandb_api_key \
    --disable_wandb $disable_wandb \
    --data_folder $data_folder \
    --unimol_feature_folder $unimol_feature_folder \
    --dataset_name "$dataset_name" \
    --checkpoint_path ./models/unimol_base.pt \
    --num_workers $num_workers \
    --num_preprocess_workers $num_preprocess_workers \
    --pin_memory $pin_memory \
    --ignore_preprocessed_dataset $ignore_preprocessed_dataset \
    --uncertainty_method $uncertainty_method \
    --retrain_model $retrain_model \
    --binary_classification_with_softmax $binary_classification_with_softmax \
    --regression_with_variance $regression_with_variance \
    --lr $lr \
    --n_epochs $n_epochs \
    --valid_tolerance $valid_tolerance \
    --batch_size $batch_size \
    --batch_size_inference $batch_size_inference \
    --seed $seed \
    --n_test $n_test
done
