
# --- toggle the following command to decide whether train the model on each dataset ---
# --- the argument values do not matter ---

# regression tasks
train_on_esol=true
train_on_freesolv=true
train_on_lipo=true
train_on_qm7=true
# train_on_qm8=true
# train_on_qm9=true

# single-task classification
train_on_bbbp=true
train_on_bace=true
train_on_hiv=true

# multi-task classification
train_on_tox21=true
train_on_toxcast=true
train_on_clintox=true
train_on_sider=true
# train_on_muv=true
# train_on_pcba=true

# --- dataset choosing region ends ---

# --- universal arguments ---
cuda_device=$1

wandb_api_key="86efcc8aa38a82bd8128db4c1cee3acde0f33920"
disable_wandb=false

data_folder="./data/files"
feature_type="rdkit"
n_feature_generating_threads=8
ignore_preprocessed_dataset=false

uncertainty_method="none"  # this is subject to change
retrain_model=false

binary_classification_with_softmax=false
regression_with_variance=true

batch_size=256
n_epochs=100
# --- universal arguments region ends ---

# --- training scripts ---
if [ -z ${train_on_bbbp+x} ];
then
  echo "skip bbbp"
else
  CUDA_VISIBLE_DEVICES=$cuda_device python run_base.py \
    --wandb_api_key $wandb_api_key \
    --disable_wandb $disable_wandb \
    --data_folder $data_folder \
    --dataset_name bbbp \
    --feature_type $feature_type \
    --n_feature_generating_threads $n_feature_generating_threads \
    --ignore_preprocessed_dataset $ignore_preprocessed_dataset \
    --uncertainty_method $uncertainty_method \
    --retrain_model $retrain_model \
    --binary_classification_with_softmax $binary_classification_with_softmax \
    --regression_with_variance $regression_with_variance \
    --n_epochs $n_epochs \
    --batch_size $batch_size
fi

if [ -z ${train_on_bace+x} ];
then
  echo "skip bace"
else
  CUDA_VISIBLE_DEVICES=$cuda_device python run_base.py \
    --wandb_api_key $wandb_api_key \
    --disable_wandb $disable_wandb \
    --data_folder $data_folder \
    --dataset_name bace \
    --feature_type $feature_type \
    --n_feature_generating_threads $n_feature_generating_threads \
    --ignore_preprocessed_dataset $ignore_preprocessed_dataset \
    --uncertainty_method $uncertainty_method \
    --retrain_model $retrain_model \
    --binary_classification_with_softmax $binary_classification_with_softmax \
    --regression_with_variance $regression_with_variance \
    --n_epochs $n_epochs \
    --batch_size $batch_size
fi

if [ -z ${train_on_hiv+x} ];
then
  echo "skip hiv"
else
  CUDA_VISIBLE_DEVICES=$cuda_device python run_base.py \
    --wandb_api_key $wandb_api_key \
    --disable_wandb $disable_wandb \
    --data_folder $data_folder \
    --dataset_name hiv \
    --feature_type $feature_type \
    --n_feature_generating_threads $n_feature_generating_threads \
    --ignore_preprocessed_dataset $ignore_preprocessed_dataset \
    --uncertainty_method $uncertainty_method \
    --retrain_model $retrain_model \
    --binary_classification_with_softmax $binary_classification_with_softmax \
    --regression_with_variance $regression_with_variance \
    --n_epochs $n_epochs \
    --batch_size $batch_size
fi

if [ -z ${train_on_tox21+x} ];
then
  echo "skip tox21"
else
  CUDA_VISIBLE_DEVICES=$cuda_device python run_base.py \
    --wandb_api_key $wandb_api_key \
    --disable_wandb $disable_wandb \
    --data_folder $data_folder \
    --dataset_name tox21 \
    --feature_type $feature_type \
    --n_feature_generating_threads $n_feature_generating_threads \
    --ignore_preprocessed_dataset $ignore_preprocessed_dataset \
    --uncertainty_method $uncertainty_method \
    --retrain_model $retrain_model \
    --binary_classification_with_softmax $binary_classification_with_softmax \
    --regression_with_variance $regression_with_variance \
    --n_epochs $n_epochs \
    --batch_size $batch_size
fi

if [ -z ${train_on_toxcast+x} ];
then
  echo "skip toxcast"
else
  CUDA_VISIBLE_DEVICES=$cuda_device python run_base.py \
    --wandb_api_key $wandb_api_key \
    --disable_wandb $disable_wandb \
    --data_folder $data_folder \
    --dataset_name toxcast \
    --feature_type $feature_type \
    --n_feature_generating_threads $n_feature_generating_threads \
    --ignore_preprocessed_dataset $ignore_preprocessed_dataset \
    --uncertainty_method $uncertainty_method \
    --retrain_model $retrain_model \
    --binary_classification_with_softmax $binary_classification_with_softmax \
    --regression_with_variance $regression_with_variance \
    --n_epochs $n_epochs \
    --batch_size $batch_size
fi

if [ -z ${train_on_clintox+x} ];
then
  echo "skip clintox"
else
  CUDA_VISIBLE_DEVICES=$cuda_device python run_base.py \
    --wandb_api_key $wandb_api_key \
    --disable_wandb $disable_wandb \
    --data_folder $data_folder \
    --dataset_name clintox \
    --feature_type $feature_type \
    --n_feature_generating_threads $n_feature_generating_threads \
    --ignore_preprocessed_dataset $ignore_preprocessed_dataset \
    --uncertainty_method $uncertainty_method \
    --retrain_model $retrain_model \
    --binary_classification_with_softmax $binary_classification_with_softmax \
    --regression_with_variance $regression_with_variance \
    --n_epochs $n_epochs \
    --batch_size $batch_size
fi

if [ -z ${train_on_sider+x} ];
then
  echo "skip sider"
else
  CUDA_VISIBLE_DEVICES=$cuda_device python run_base.py \
    --wandb_api_key $wandb_api_key \
    --disable_wandb $disable_wandb \
    --data_folder $data_folder \
    --dataset_name sider \
    --feature_type $feature_type \
    --n_feature_generating_threads $n_feature_generating_threads \
    --ignore_preprocessed_dataset $ignore_preprocessed_dataset \
    --uncertainty_method $uncertainty_method \
    --retrain_model $retrain_model \
    --binary_classification_with_softmax $binary_classification_with_softmax \
    --regression_with_variance $regression_with_variance \
    --n_epochs $n_epochs \
    --batch_size $batch_size
fi

if [ -z ${train_on_muv+x} ];
then
  echo "skip muv"
else
  CUDA_VISIBLE_DEVICES=$cuda_device python run_base.py \
    --wandb_api_key $wandb_api_key \
    --disable_wandb $disable_wandb \
    --data_folder $data_folder \
    --dataset_name muv \
    --feature_type $feature_type \
    --n_feature_generating_threads $n_feature_generating_threads \
    --ignore_preprocessed_dataset $ignore_preprocessed_dataset \
    --uncertainty_method $uncertainty_method \
    --retrain_model $retrain_model \
    --binary_classification_with_softmax $binary_classification_with_softmax \
    --regression_with_variance $regression_with_variance \
    --n_epochs $n_epochs \
    --batch_size $batch_size
fi

if [ -z ${train_on_pcba+x} ];
then
  echo "skip pcba"
else
  CUDA_VISIBLE_DEVICES=$cuda_device python run_base.py \
    --wandb_api_key $wandb_api_key \
    --disable_wandb $disable_wandb \
    --data_folder $data_folder \
    --dataset_name pcba \
    --feature_type $feature_type \
    --n_feature_generating_threads $n_feature_generating_threads \
    --ignore_preprocessed_dataset $ignore_preprocessed_dataset \
    --uncertainty_method $uncertainty_method \
    --retrain_model $retrain_model \
    --binary_classification_with_softmax $binary_classification_with_softmax \
    --regression_with_variance $regression_with_variance \
    --n_epochs $n_epochs \
    --batch_size $batch_size
fi

if [ -z ${train_on_esol+x} ];
then
  echo "skip esol"
else
  CUDA_VISIBLE_DEVICES=$cuda_device python run_base.py \
    --wandb_api_key $wandb_api_key \
    --disable_wandb $disable_wandb \
    --data_folder $data_folder \
    --dataset_name esol \
    --feature_type $feature_type \
    --n_feature_generating_threads $n_feature_generating_threads \
    --ignore_preprocessed_dataset $ignore_preprocessed_dataset \
    --uncertainty_method $uncertainty_method \
    --retrain_model $retrain_model \
    --binary_classification_with_softmax $binary_classification_with_softmax \
    --regression_with_variance $regression_with_variance \
    --n_epochs $n_epochs \
    --batch_size $batch_size
fi

if [ -z ${train_on_freesolv+x} ];
then
  echo "skip freesolv"
else
  CUDA_VISIBLE_DEVICES=$cuda_device python run_base.py \
    --wandb_api_key $wandb_api_key \
    --disable_wandb $disable_wandb \
    --data_folder $data_folder \
    --dataset_name freesolv \
    --feature_type $feature_type \
    --n_feature_generating_threads $n_feature_generating_threads \
    --ignore_preprocessed_dataset $ignore_preprocessed_dataset \
    --uncertainty_method $uncertainty_method \
    --retrain_model $retrain_model \
    --binary_classification_with_softmax $binary_classification_with_softmax \
    --regression_with_variance $regression_with_variance \
    --n_epochs $n_epochs \
    --batch_size $batch_size
fi

if [ -z ${train_on_lipo+x} ];
then
  echo "skip lipo"
else
  CUDA_VISIBLE_DEVICES=$cuda_device python run_base.py \
    --wandb_api_key $wandb_api_key \
    --disable_wandb $disable_wandb \
    --data_folder $data_folder \
    --dataset_name lipo \
    --feature_type $feature_type \
    --n_feature_generating_threads $n_feature_generating_threads \
    --ignore_preprocessed_dataset $ignore_preprocessed_dataset \
    --uncertainty_method $uncertainty_method \
    --retrain_model $retrain_model \
    --binary_classification_with_softmax $binary_classification_with_softmax \
    --regression_with_variance $regression_with_variance \
    --n_epochs $n_epochs \
    --batch_size $batch_size
fi

if [ -z ${train_on_qm7+x} ];
then
  echo "skip qm7"
else
  CUDA_VISIBLE_DEVICES=$cuda_device python run_base.py \
    --wandb_api_key $wandb_api_key \
    --disable_wandb $disable_wandb \
    --data_folder $data_folder \
    --dataset_name qm7 \
    --feature_type $feature_type \
    --n_feature_generating_threads $n_feature_generating_threads \
    --ignore_preprocessed_dataset $ignore_preprocessed_dataset \
    --uncertainty_method $uncertainty_method \
    --retrain_model $retrain_model \
    --binary_classification_with_softmax $binary_classification_with_softmax \
    --regression_with_variance $regression_with_variance \
    --n_epochs $n_epochs \
    --batch_size $batch_size
fi

if [ -z ${train_on_qm8+x} ];
then
  echo "skip qm8"
else
  CUDA_VISIBLE_DEVICES=$cuda_device python run_base.py \
    --wandb_api_key $wandb_api_key \
    --disable_wandb $disable_wandb \
    --data_folder $data_folder \
    --dataset_name qm8 \
    --feature_type $feature_type \
    --n_feature_generating_threads $n_feature_generating_threads \
    --ignore_preprocessed_dataset $ignore_preprocessed_dataset \
    --uncertainty_method $uncertainty_method \
    --retrain_model $retrain_model \
    --binary_classification_with_softmax $binary_classification_with_softmax \
    --regression_with_variance $regression_with_variance \
    --n_epochs $n_epochs \
    --batch_size $batch_size
fi

if [ -z ${train_on_qm9+x} ];
then
  echo "skip qm9"
else
  CUDA_VISIBLE_DEVICES=$cuda_device python run_base.py \
    --wandb_api_key $wandb_api_key \
    --disable_wandb $disable_wandb \
    --data_folder $data_folder \
    --dataset_name qm9 \
    --feature_type $feature_type \
    --n_feature_generating_threads $n_feature_generating_threads \
    --ignore_preprocessed_dataset $ignore_preprocessed_dataset \
    --uncertainty_method $uncertainty_method \
    --retrain_model $retrain_model \
    --binary_classification_with_softmax $binary_classification_with_softmax \
    --regression_with_variance $regression_with_variance \
    --n_epochs $n_epochs \
    --batch_size $batch_size
fi
