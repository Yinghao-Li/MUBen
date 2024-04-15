# Running Experiments on CLI 

!!! info
    Please ensure you have clone the repo and navigated to the `MUBen/` directory in your local environment to begin working with the project.

The most straightforward approach to replicate our experimental results is using the python scripts provided in the MUBen repo with the following pipeline.

## Fine-Tuning the Models

The [./run/](https://github.com/Yinghao-Li/MUBen/tree/main/run) directory contains the entry scripts to fine-tuning each of the backbone-UQ combinations.
Currently, the script [./run/run.py](https://github.com/Yinghao-Li/MUBen/blob/main/run/run.py) is adopted to run all backbone models except for GROVER and Uni-Mol, whose entry scripts are [./run/grover.py](https://github.com/Yinghao-Li/MUBen/blob/main/run/grover.py) and [./run/unimol.py](https://github.com/Yinghao-Li/MUBen/blob/main/run/unimol.py), respectively.

### Specify Arguments Using Command Lines

An example of running the **DNN** model with **RDKit** features with the **MC Dropout** UQ method on the **BBBP** dataset is
```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="." \
python ./run/run.py \
    --descriptor_type "RDKit" \
    --data_folder "./data/files" \
    --dataset_name "bbbp" \
    --uncertainty_method "MCDropout" \
    --lr 0.0001 \
    --n_epochs 200 \
    --batch_size 256 \
    --seed 0
```
In the example, the `--descriptor_type` argument is used to select the backbone models used in our experiments.
It has 4 options: {"RDKit", "Linear", "2D", "3D"}, which corresponds to the DNN, ChemBERTa, GIN and TorchMD-NET backbone models in the CLI, respectively.
In the future versions, we may consider including multiple backbone models that correspond to one descriptor, which requires us to specify the `--model_name` argument to separate the backbones.
But currently, we do not need to worry about that and can leave `--model_name` as default.

!!! info
    For the interpretation of each argument, please check the [`muben.args` API](./muben.args.md) or directly refer to the [code implementation](https://github.com/Yinghao-Li/MUBen/tree/main/muben/args).
    Notice that the API documentation may not be entirely comprehensive.

Similarly, we can also run the **ChemBERTa** model with the **SGLD** UQ method on the **ESOL** dataset using
```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="." \
python ./run/run.py \
    --descriptor_type "Linear" \
    --data_folder "./data/files" \
    --dataset_name "esol" \
    --uncertainty_method "SGLD" \
    --lr 0.00005 \
    --n_epochs 100 \
    --batch_size 128 \
    --seed 0 \
    --regression_with_variance
```
!!! warning
    For regression tasks, the argument `--regression_with_variance` is vital to guarantee a valid result with predicted variance.

To run GROVER or Uni-Mol, we just need to replace `run.py` by the corresponding script, and slightly modify the arguments:
```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="." \
python ./run/unimol.py \
    --data_folder "./data/files" \
    --unimol_feature_folder "./data/UniMol/" \
    --dataset_name "bbbp" \
    --checkpoint_path "./models/unimol_base.pt" \
    --uncertainty_method "MCDropout" \
    ...
```

### Specify Arguments using `.yaml` Files

Another way of specifying arguments is through the `.yaml` scripts, which provides more readable data structure than `.json` files.
We have provided an [example configuration script](https://github.com/Yinghao-Li/MUBen/blob/main/scripts/config-example.yaml) within the [./scripts/](https://github.com/Yinghao-Li/MUBen/tree/main/scripts) directory, which runs GIN on the FreeSolv dataset with deterministic ("none") UQ method.
To use it to specify arguments, we can run the python program with
```bash
PYTHONPATH="." CUDA_VISIBLE_DEVICES=0 python ./run/run.py ./scripts/config-example.yaml
```
This approach could be helpful while debugging the code on VSCode.


## Logging and WandB

By default, this project uses local logging files (`*.log`) and [WandB](https://wandb.ai/site) to track training status.

The log files are stored as `./logs/<dataset>/<model>/<uncertainty>/<running_time>.log`.
You can change the file path by specifying the `--log_path` argument, or disable log saving by setting `--log_path="disabled"`.

To use WandB, you first need to register an account and sign in on your machine with `wandb login`.
If you are running your code on a public device, you can instead use program-wise signing in by specifying the `--wandb_api_key` argument while running our code.
You can find your API key in your browser here: https://wandb.ai/authorize.
To disable WandB, use `--disable_wandb [true]`.
By default, we use `MUBen-<dataset>` as WandB project name and `<model>-<uncertainty>` as the model name.
You can change this behavior by specifying the `--wandb_project` and `--wandb_name` arguments.

## Data Loading

The progress will automatically create the necessary features (molecular descriptors) required by backbone models from the SMILES strings if they are loaded properly.
The processed features are stored in the `<bottom-level data folder>/processed/` directory as `<train/valid/test>.pt` files by default, and will be automatically loaded the next time you apply the same backbone model on the same dataset.
You can change this behavior with `--disable_dataset_saving` for disabling dataset saving or `--ignore_preprocessed_dataset` for not loading from the saved (processed) dataset.

Constructing Morgan fingerprint, RDKit features or 3D conformations for Uni-Mol may take a while.
You can accelerate this process by utilizing multiple threads `--num_preprocess_workers=n>1` (default is 8).
For 3D conformations, we directly take advantage of the results from Uni-Mol but still keep the choice of generating them by ourselves if the Uni-Mol data files are not found.

## Calculating Metrics

During training, we only calculate metrics necessary for early stopping and simple prediction performance evaluation.
To get other metrics, you need to use the `./assist/results_get_metrics.py` file.

Specifically, you need to save the model predictions by **not** setting `--disable_dataset_saving`.
The results are saved as `./<result_folder>/<dataset_name>/<model_name>/<uncertainty_method>/seed-<seed>/preds/<test_idx>.pt` files.
When the training is finished, you can run the `./assist/results_get_metrics.py` file to generate all metrics for your model predictions.
For example:
```bash
PYTHONPATH="." python ./assist/results_get_metrics.py [arguments]
```
Make sure the arguments are updated to your needs.
