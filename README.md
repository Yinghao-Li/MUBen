# MUBen: **M**olecular **U**ncertainty **Ben**mark
Code associated with paper *MUBen: Benchmarking the Uncertainty of Pre-Trained Models for Molecular Property Prediction*.

The code is built to expose implementation details as much as possible and be easily extendable.
Questions and suggestions are welcome if you find it hard to use our code.

## 1. DATA

We utilize the datasets prepared by [Uni-Mol](https://github.com/dptech-corp/Uni-Mol/tree/main/unimol).
You find the data [here](https://github.com/dptech-corp/Uni-Mol/tree/main/unimol#:~:text=pockets.tar.gz-,molecular%20property,-3.506GB) or directly download it through [this link](https://bioos-hermite-beijing.tos-cn-beijing.volces.com/unimol_data/finetune/molecular_property_prediction.tar.gz).
We place the unzipped files into `<workspaceFolder>/data/UniMol` by default.
For convenience, you are suggested to rename the `qm7dft`, `qm8dft`, and `qm9dft` folders to `qm7`, `qm8`, and `qm9`.

Afterwards, you can transfer the dataset format into ours by running
```bash
PYTHONPATH="." python ./assist/build_dataset_from_unimol.py
``` 
suppose you are at the project root directory.
You can specify the input (Uni-Mol) and output data directories with `--unimol_data_dir` and `--output_dir` arguments.
The script will convert *all* datasets by default (excluding PCBA).
If you want to specify a subset of datasets, you can specify the argument `--dataset_names` with the target dataset names with lowercase letters.

**Notice**: If you would like to run the Uni-Mol model, you need to keep the original `UniMol` data as we will use the pre-defined molecule conformations.
Otherwise, it is safe to remove the original data.

### Options

If you do not want to use Uni-Mol data, you can try the scripts within the `legacy` folder, including `build_dgllife_datasets.py`, and `build_qm[7,8,9]_dataset.py`.
Notice that this may result in training/validation/test partitions different from what is being used in our experiments.

## 2. REQUIREMENTS

Please find the required packages in `requirements.txt`.
Our code is developed with `Python 3.10` and may not work with Python versions earlier than `3.9`.
It is recommended to create a new `conda` environment with

```bash
conda create --name <env_name> --file requirements.txt
```

### External Dependencies

The backbone models `GROVER` and `Uni-Mol` requires loading pre-trained model checkpoints.

- The `GROVER-base` checkpoint is available at GROVER's [project repo](https://github.com/tencent-ailab/grover) or can be directly downloaded through [this link](https://ai.tencent.com/ailab/ml/ml-data/grover-models/pretrain/grover_base.tar.gz).
Unzip the downloaded `.tar.gz` file to get the `.pt` checkpoint.
- The `Uni-Mol` checkpoint is available at Uni-Mol's [project repo](https://github.com/dptech-corp/Uni-Mol/tree/main/unimol) or can be directly downloaded through [this link](https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt).

By default, the code will look for the models at locations `<project root>/models/grover_base.pt` and `<project root>/models/unimol_base.pt`, respectively.
You need to specify the `--checkpoint_path` argument if you prefer other locations and checkpoint names.

## 3. RUN

To run each of the four backbone models with uncertainty estimation methods, you can check the `run_*.py` files in the root directory.
Example shell scripts are provided in the `./scripts` folder as `.sh` files.
You can use them through
```bash
./scripts/run_dnn_rdkit.sh <CUDA_VISIBLE_DEVICES>
```
as an example.

To get a detailed description of each argument, you can use the `--help` argument:
```bash
python run_dnn.py --help
```

## 4. CITATION

Coming soon.
